"""Extract search filters (date range, topic) from a natural-language question with an LLM."""

import logging
from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from app.config import get_settings
from app.schemas import SearchFilters
from app.services.llm import _client
from app.topics import TOPICS

logger = logging.getLogger(__name__)

CORPUS_START = date(1987, 2, 26)
CORPUS_END = date(1987, 10, 20)

TopicCode = Literal[tuple(TOPICS)]  # type: ignore[valid-type]


class ExtractedFilters(BaseModel):
    """Response format the LLM must follow. Every field is required but nullable (strict mode)."""

    date_from: date | None = Field(description="First day the question refers to, or null")
    date_to: date | None = Field(description="Last day the question refers to, or null")
    # Generated before `topic` on purpose: with only the enum, the model starts typing a word
    # like "oil" and constrained decoding completes it to the nearest code ("oilseed").
    subject: str = Field(description="The question's subject in a few plain words")
    topic: TopicCode | None = Field(description="Single best matching topic code, or null")


SYSTEM_PROMPT = f"""You turn questions about a news archive into search filters.
The archive holds Reuters business news from {CORPUS_START} to {CORPUS_END}.

Dates:
- Set date_from/date_to only if the question refers to a specific day, week, month or period.
- A single day: use that day for both fields. A month: its first and last day.
- "Early", "mid" and "late" in a month: days 1-10, 11-20 and 21 to month end.
- If no year is given, assume 1987.
- Leave both null for vague or relative references you cannot pin down ("recently", "last week").

Topic:
- Set a topic only if the question is clearly about one of the subjects below; otherwise null.
- Pick the most specific matching code. Match on meaning, not on shared words:
  "oil prices", "OPEC", "oil output" -> crude; "takeover", "merger" -> acq;
  "profits", "quarterly results" -> earn; "dollar against the yen" -> money-fx.

Topics (code: description):
""" + "\n".join(f"{code}: {description}" for code, description in TOPICS.items())


async def extract_filters(question: str) -> ExtractedFilters | None:
    """Ask the LLM for filters. Returns None if the call or validation fails, so search still runs."""
    try:
        completion = await _client().chat.completions.parse(
            model=get_settings().filter_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            response_format=ExtractedFilters,
        )
        return completion.choices[0].message.parsed
    except Exception:
        logger.exception("Filter extraction failed; searching without extracted filters")
        return None


def merge_filters(explicit: SearchFilters, extracted: ExtractedFilters | None) -> SearchFilters:
    """Combine client-supplied and extracted filters. Explicit values always win.

    Dates are taken as a pair: if the client set either bound, extracted dates are ignored.
    Extracted single days are widened by one day, since news about an event often runs
    the day after it.
    """
    if extracted is None:
        return explicit

    date_from, date_to = explicit.date_from, explicit.date_to
    if date_from is None and date_to is None:
        date_from, date_to = extracted.date_from, extracted.date_to
        if date_from and date_to and date_from == date_to:
            date_to += timedelta(days=1)

    try:
        return SearchFilters(date_from=date_from, date_to=date_to, topic=explicit.topic or extracted.topic)
    except ValidationError:
        # e.g. the LLM returned date_from after date_to; keep only what is certainly valid.
        return SearchFilters(
            date_from=explicit.date_from,
            date_to=explicit.date_to,
            topic=explicit.topic or extracted.topic,
        )
