"""Chainlit chat UI on top of the /query endpoint.

Mounted at /ui by app.main. It talks to the API over HTTP like any other client, so it can
also run on its own: `python -m chainlit run app/ui.py --port 8001` (with the API running at
API_URL; Chainlit defaults to port 8000 too, so leave that one to the API).
"""

import json
from datetime import datetime, timedelta

import chainlit as cl
import httpx
from chainlit.input_widget import DatePicker, Select, Slider, Switch
from httpx_sse import aconnect_sse

from app.config import get_settings
from app.topics import TOPICS

# Radix Select (used by Chainlit) rejects an empty-string value, so "no topic" needs a sentinel.
ANY_TOPIC = "any"
CORPUS_START, CORPUS_END = "1987-02-26", "1987-10-20"

SETTINGS = [
    Switch(
        id="auto_filters",
        label="Extract filters from the question",
        initial=True,
        description="Date range and topic set below take precedence.",
    ),
    DatePicker(
        id="dates",
        label="Date range",
        mode="range",
        min_date=CORPUS_START,
        max_date=CORPUS_END,
        placeholder="Any date",
    ),
    Select(
        id="topic",
        label="Topic",
        items={"Any topic": ANY_TOPIC} | {f"{name} ({code})": code for code, name in TOPICS.items()},
        initial_value=ANY_TOPIC,
    ),
    Slider(id="top_k", label="Passages to retrieve", initial=get_settings().top_k, min=1, max=20, step=1),
]


def _picker_date(value: str) -> str:
    # The picker sends local midnight as a UTC timestamp (1987-10-19 in CEST arrives as
    # 1987-10-18T22:00:00.000Z). Shifting by 12h before truncating recovers the calendar day
    # for any UTC offset within ±12h.
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return (moment + timedelta(hours=12)).date().isoformat()


def build_payload(question: str, settings: dict) -> dict:
    payload = {
        "question": question,
        "auto_filters": settings.get("auto_filters", True),
        "top_k": int(settings.get("top_k", get_settings().top_k)),
    }
    if dates := settings.get("dates"):
        payload["date_from"], payload["date_to"] = (_picker_date(d) for d in dates)
    if (topic := settings.get("topic")) and topic != ANY_TOPIC:
        payload["topic"] = topic
    return payload


def describe_filters(applied: dict) -> str:
    parts = []
    if applied["date_from"] or applied["date_to"]:
        parts.append(f"{applied['date_from'] or '…'} → {applied['date_to'] or '…'}")
    if topic := applied["topic"]:
        parts.append(f"{TOPICS.get(topic, topic)} (`{topic}`)")
    return ", ".join(parts) or "None, searching the whole corpus."


def source_elements(sources: list[dict], message_id: str) -> list[cl.Text]:
    # Naming each element after its citation marker makes "[1]" in the answer a clickable link.
    return [
        cl.Text(
            name=f"[{i}]",
            content=f"**{chunk['metadata'].get('title', 'Untitled')}**  \n"
            f"{chunk['metadata'].get('date', '')} · distance {chunk['distance']:.3f}\n\n{chunk['text']}",
            display="side",
            for_id=message_id,
        )
        for i, chunk in enumerate(sources, start=1)
    ]


@cl.set_starters
async def set_starters(user=None, language=None) -> list[cl.Starter]:
    return [
        cl.Starter(label=question, message=question)
        for question in (
            "What happened to oil prices on October 19, 1987?",
            "Which takeovers were announced in March?",
            "How did the dollar trade against the yen in early April 1987?",
        )
    ]


@cl.on_chat_start
async def on_chat_start() -> None:
    settings = await cl.ChatSettings(SETTINGS).send()
    cl.user_session.set("settings", settings)


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    cl.user_session.set("settings", settings)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    payload = build_payload(message.content, cl.user_session.get("settings") or {})
    answer = cl.Message(content="")
    sources: list[dict] = []

    try:
        async with httpx.AsyncClient(base_url=get_settings().api_url, timeout=None) as client:
            async with aconnect_sse(client, "POST", "/query", json=payload) as events:
                if events.response.is_error:
                    await events.response.aread()
                    await cl.ErrorMessage(f"API error {events.response.status_code}: {events.response.text}").send()
                    return
                async for event in events.aiter_sse():
                    data = json.loads(event.data)
                    if event.event == "filters":
                        async with cl.Step(name="Filters", type="retrieval") as step:
                            step.output = describe_filters(data)
                    elif event.event == "sources":
                        sources = data
                    elif event.event == "token":
                        await answer.stream_token(data)
    except httpx.HTTPError as exc:
        await cl.ErrorMessage(f"Could not reach the API at {get_settings().api_url}: {exc}").send()
        return

    if sources:
        answer.elements = source_elements(sources, answer.id)
        answer.content += "\n\n**Sources:** " + " ".join(f"[{i}]" for i in range(1, len(sources) + 1))
    await answer.send()
