"""Review UI for the golden dataset.

Usage (from the repo root, after `pip install -r requirements-dev.txt`):
    .venv/bin/streamlit run evals/review_app.py

Shows one case at a time: the editable fields on the left, the full text of the expected
articles on the right with the key facts highlighted, and below them the articles the latest
evaluation runs retrieved, each with a button to add it to expected_doc_ids.
Saving writes straight back to evals/golden.jsonl, so `git diff` shows what the review changed.
"""

import html
import json
import os
import re
import sys
from pathlib import Path

import streamlit as st

# Streamlit runs this file as a script: make the repo root importable and the working directory,
# since the dataset, runs and corpus paths are relative to it.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from evals.generate import DEFAULT_OUT as GOLDEN, load_cases  # noqa: E402
from scripts.ingest_reuters import download_archive, iter_articles  # noqa: E402

RUNS_DIR = Path("evals/runs")
STYLES = ["dated", "undated"]

st.set_page_config(page_title="Golden dataset review", layout="wide")


@st.cache_resource(show_spinner="Reading the Reuters archive...")
def articles() -> dict[str, tuple[str, dict]]:
    download_archive()
    return {doc_id: (content, metadata) for doc_id, content, metadata in iter_articles()}


def load_runs(auto_filters: bool) -> list[tuple[str, dict[str, dict]]]:
    """All evaluation runs with this auto_filters setting, newest first, results keyed by case id."""
    runs = []
    for path in sorted(RUNS_DIR.glob("*-retrieval.json"), reverse=True):
        run = json.loads(path.read_text())
        if run["config"]["auto_filters"] == auto_filters:
            runs.append((path.name, {r["id"]: r for r in run["results"]}))
    return runs


def across_runs(results: list[dict]) -> str:
    """One line summarizing how a case did over several runs, e.g. 'rank 1 in 3 of 4 runs'."""
    n = len(results)
    parts = [f"rank 1 in {sum(r['rank'] == 1 for r in results)} of {n} run{'s' if n > 1 else ''}"]
    if lower := [r["rank"] for r in results if r["rank"] and r["rank"] > 1]:
        parts.append(f"lower rank in {len(lower)} ({', '.join(map(str, lower))})")
    if filtered := [r for r in results if r["filtered_out"]]:
        topics = sorted({r["applied_filters"].get("topic") or "dates" for r in filtered})
        parts.append(f"filtered out in {len(filtered)} ({', '.join(topics)})")
    if missed := sum(r["rank"] is None and not r["filtered_out"] for r in results):
        parts.append(f"not retrieved in {missed}")
    return "; ".join(parts)


def filter_check(applied: dict, metadata: dict) -> str:
    """Compare the applied filters with an article's date and labels, e.g. 'topic acq vs earn ✗'."""
    parts = []
    date = metadata.get("date")
    date_from, date_to = applied.get("date_from"), applied.get("date_to")
    if date_from or date_to:
        ok = bool(date) and (not date_from or date >= date_from) and (not date_to or date <= date_to)
        parts.append(f"date {date_from or '…'} – {date_to or '…'} vs {date or 'none'} {'✓' if ok else '✗'}")
    if topic := applied.get("topic"):
        labels = metadata.get("topics", [])
        parts.append(f"topic {topic} vs {', '.join(labels) or 'none'} {'✓' if topic in labels else '✗'}")
    return " · ".join(parts) or "no filters applied"


def normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def highlight(text: str, facts: list[str]) -> str:
    """Article text as HTML with each key fact marked, ignoring case and line breaks."""
    escaped = html.escape(text)
    for fact in sorted(facts, key=len, reverse=True):
        tokens = html.escape(fact).split()
        if tokens:
            pattern = r"\s+".join(re.escape(token) for token in tokens)
            escaped = re.sub(pattern, lambda m: f"<mark>{m.group(0)}</mark>", escaped, flags=re.IGNORECASE)
    style = "white-space: pre-wrap; font-family: ui-monospace, monospace; font-size: 0.85rem; line-height: 1.45"
    return f'<div style="{style}">{escaped}</div>'


def lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def write_cases(cases: list[dict]) -> None:
    tmp = GOLDEN.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(case) + "\n" for case in cases))
    tmp.replace(GOLDEN)  # atomic: a crash mid-write never leaves a half-written dataset


# --- state helpers -------------------------------------------------------------------------

state = st.session_state


def init_fields(case: dict) -> None:
    """Load a case into the widget state the first time it is shown."""
    cid = case["id"]
    defaults = {
        f"question_{cid}": case["question"],
        f"style_{cid}": case.get("style") or "undated",
        f"answer_{cid}": case.get("reference_answer", ""),
        f"facts_{cid}": "\n".join(case.get("key_facts", [])),
        f"expected_{cid}": "\n".join(case["expected_doc_ids"]),
        f"notes_{cid}": case.get("notes", ""),
    }
    for key, value in defaults.items():
        state.setdefault(key, value)


def save(cid: str, reviewed: bool | None = None, next_id: str | None = None) -> None:
    expected = list(dict.fromkeys(lines(state[f"expected_{cid}"])))
    unknown = [doc_id for doc_id in expected if doc_id not in articles()]
    if not expected or unknown:
        state.flash = ("error", f"Not saved: unknown article ids {unknown}" if unknown else "Not saved: no expected article")
        return

    cases = load_cases(GOLDEN)
    case = next(c for c in cases if c["id"] == cid)
    case["question"] = state[f"question_{cid}"].strip()
    case["style"] = state[f"style_{cid}"]
    case["reference_answer"] = state[f"answer_{cid}"].strip()
    case["key_facts"] = lines(state[f"facts_{cid}"])
    case["expected_doc_ids"] = expected
    if notes := state[f"notes_{cid}"].strip():
        case["notes"] = notes
    else:
        case.pop("notes", None)
    if reviewed is not None:
        case["reviewed"] = reviewed
    write_cases(cases)

    state.flash = ("success", f"Saved {cid}" + (" as reviewed" if reviewed else ""))
    if next_id:
        state.case_id = next_id


def add_expected(cid: str, doc_id: str) -> None:
    key = f"expected_{cid}"
    if doc_id not in lines(state[key]):
        state[key] = state[key].rstrip() + "\n" + doc_id


# --- layout --------------------------------------------------------------------------------

cases = load_cases(GOLDEN)
runs = {label: load_runs(auto) for label, auto in (("with auto filters", True), ("without auto filters", False))}


def is_miss(cid: str) -> bool:
    """Not ranked first in the latest run of either setting."""
    return any(r and cid in r[0][1] and r[0][1][cid]["rank"] != 1 for r in runs.values())


with st.sidebar:
    st.header("Golden dataset")
    n_reviewed = sum(bool(c.get("reviewed")) for c in cases)
    st.progress(n_reviewed / len(cases) if cases else 0.0, text=f"{n_reviewed} of {len(cases)} reviewed")
    view = st.radio("Show", ["Unreviewed", "Misses in latest runs", "All"])
    shown = [
        c for c in cases
        if view == "All"
        or (view == "Unreviewed" and not c.get("reviewed"))
        or (view == "Misses in latest runs" and is_miss(c["id"]))
    ]
    if not shown:
        st.info("Nothing to show.")
        st.stop()
    ids = [c["id"] for c in shown]
    if state.get("case_id") not in ids:
        state.case_id = ids[0]
    labels = {c["id"]: f"{'✅' if c.get('reviewed') else '⬜'} {c['id']}{'  ⚠️ miss' if is_miss(c['id']) else ''}" for c in shown}
    st.selectbox("Case", ids, key="case_id", format_func=labels.get)
    st.caption("✅ reviewed · ⚠️ not ranked first in a latest run")
    for label, label_runs in runs.items():
        st.caption(f"Runs {label}: {len(label_runs)}, latest {label_runs[0][0] if label_runs else 'none'}")

case = next(c for c in cases if c["id"] == state.case_id)
cid = case["id"]
init_fields(case)
position = ids.index(cid)
next_id = ids[position + 1] if position + 1 < len(ids) else None

if flash := state.pop("flash", None):
    getattr(st, flash[0])(flash[1])

with st.expander("How to review: judge the case, not the system", expanded=not case.get("reviewed")):
    st.markdown(
        "Check the **case**: is the question realistic, specific and about one thing? Does the expected "
        "article answer it? Is the reference answer correct and complete? Is every key fact in the article "
        "and essential? Does the style match?\n\n"
        "The **run results** below only help with two decisions: whether a retrieved article answers the "
        "question equally well (then add it to the expected articles), and whether a miss is the system's "
        "fault. A wrong filter or a topic that differs from the article's Reuters labels is the system's "
        "fault: **never change a case because of it**, and no note is needed. The Reuters labels are fixed "
        "corpus data; they are not reviewed."
    )

st.subheader(f"{cid} · {'reviewed' if case.get('reviewed') else 'not reviewed'}")
left, right = st.columns([2, 3], gap="large")

with left:
    st.text_input("Question", key=f"question_{cid}")
    st.radio("Style", STYLES, key=f"style_{cid}", horizontal=True,
             help="dated: mentions when the news was reported. undated: mentions no date or period.")
    st.text_area("Reference answer", key=f"answer_{cid}", height=110)
    st.text_area("Key facts (one per line)", key=f"facts_{cid}", height=90)

    source_text = normalize(" ".join(articles()[d][0] for d in lines(state[f"expected_{cid}"]) if d in articles()))
    for fact in lines(state[f"facts_{cid}"]):
        found = normalize(fact) in source_text
        st.caption(f"{'✓' if found else '✗ not found verbatim:'} {fact}")

    st.text_area("Expected article ids (one per line)", key=f"expected_{cid}", height=90)
    st.text_area("Notes (what you changed and why)", key=f"notes_{cid}", height=70)

    b1, b2, b3 = st.columns(3)
    b1.button("Save", on_click=save, args=(cid,), use_container_width=True)
    b2.button("✅ Reviewed, next", on_click=save, args=(cid, True, next_id), type="primary", use_container_width=True)
    b3.button("Mark unreviewed", on_click=save, args=(cid, False), use_container_width=True)
    st.caption(f"Case {position + 1} of {len(ids)} in this view. Saving writes to {GOLDEN}.")

with right:
    facts = lines(state[f"facts_{cid}"])
    for i, doc_id in enumerate(lines(state[f"expected_{cid}"])):
        if doc_id not in articles():
            st.warning(f"Unknown article id: {doc_id}")
            continue
        content, metadata = articles()[doc_id]
        header = f"Expected: {doc_id} · {metadata.get('date', '?')} · {', '.join(metadata.get('topics', [])) or 'no topics'}"
        with st.expander(header, expanded=i == 0):
            st.markdown(highlight(content, facts), unsafe_allow_html=True)

st.divider()
st.markdown("#### Evaluation runs")
st.caption("The articles retrieved instead are the candidates for equally good answers: add them if they answer the question as well.")
run_columns = st.columns(len(runs), gap="large")
expected_now = set(lines(state[f"expected_{cid}"]))
for column, (label, label_runs) in zip(run_columns, runs.items()):
    with column:
        st.markdown(f"**{label}**")
        results = [run[cid] for _, run in label_runs if cid in run]
        if not results:
            st.caption("Not in any run.")
            continue
        st.markdown(f"All runs: {across_runs(results)}")

        result = results[0]
        rank = f"rank {result['rank']}" if result["rank"] else "not retrieved"
        reason = " · filtered out by the applied filters" if result["filtered_out"] else ""
        st.markdown(f"Latest run: {'✅' if result['rank'] == 1 else '⚠️'} {rank}{reason}")
        if result["applied_filters"]:
            for doc_id in result["expected_doc_ids"]:
                if doc_id in articles():
                    st.caption(f"Filters vs {doc_id}: {filter_check(result['applied_filters'], articles()[doc_id][1])}")
        for doc_id in dict.fromkeys(hit["document_id"] for hit in result["retrieved"]):
            if doc_id not in articles():
                continue
            content, metadata = articles()[doc_id]
            mark = "✓ expected · " if doc_id in expected_now else ""
            topics = ", ".join(metadata.get("topics", [])) or "no topics"
            with st.expander(f"{mark}{doc_id} · {metadata.get('date', '?')} · {topics} · {metadata['title']}"):
                st.markdown(highlight(content, facts), unsafe_allow_html=True)
                if doc_id not in expected_now:
                    st.button("Add to expected", key=f"add_{label}_{cid}_{doc_id}",
                              on_click=add_expected, args=(cid, doc_id))
