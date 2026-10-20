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
from datetime import datetime
from pathlib import Path

import streamlit as st

# Streamlit runs this file as a script: make the repo root importable and the working directory,
# since the dataset, runs and corpus paths are relative to it.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from evals import duplicates  # noqa: E402
from evals.checks import DEFAULT_OUT as GOLDEN, case_problems, content_words, load_cases, missing_facts  # noqa: E402
from evals.duplicates import MAX_DISTANCE  # noqa: E402
from scripts.ingest_reuters import download_archive, iter_articles  # noqa: E402

RUNS_DIR = Path("evals/runs")
STYLES = ["dated", "undated"]

ADD_HELP = """**Add a retrieved article to the expected articles?**

Ask: would a user asking this question be fully satisfied with this article alone?

Add it if:
- it is the same story again (an update, a repeat, a short version)
- it contains the facts of the reference answer; many highlighted key facts are a strong sign
- a user would not care which of the two they got

Don't add it if:
- it is only on the same subject
- it answers only part of the question
- it gives different figures, or another source's view

The column without auto filters is usually the better candidate list: no wrong filter has removed anything.

**Key facts n/m** counts the case's key facts found verbatim in the article.
An expected article should contain all of them.

**Distance** is the cosine distance between the question and the article's closest chunk:
lower is closer, and search returns the lowest. Compare distances within one question, not across questions.
Two copies of one story usually have almost the same distance."""

DUPLICATES_HELP = f"""**Other versions of the expected story**, found by comparing the articles' first chunks
(distance at most {MAX_DISTANCE}). Reuters often sent a story twice: a repeat, an update, or a short
highlights item next to a detailed table.

Shown first: candidates that contain **all key facts**; add one if a user would be fully satisfied with it.

Collapsed: similar articles that **lack key facts**. Usually the same kind of report with other figures,
such as last week's purchases or another company's earnings table. These are rarely duplicates.

Nothing is added automatically."""

ANSWER_HELP = """**The answer the system gave in an answer-scoring run** (`python -m evals.score_answers`).

🟨 Yellow: a key fact found verbatim, the same check as in the articles.

🟦 Blue: the judge's quote for a key fact that is not verbatim.
The judge decided the answer states the fact in other words; check that it does.

Where the two checks disagree, one of them is wrong:
- **judge ✓, verbatim ✗**: a paraphrase ('2.5 percent' for '2.5 pct'), or the judge is too lenient.
- **judge ✗, verbatim ✓**: the fact matched by accident (too short or too common), or the judge is wrong.
- **quote not in the answer**: the judge invented its evidence; don't trust that verdict.

**Groundedness**: whether every claim in the answer is supported by the retrieved passages (MLflow's
built-in judge). A grounded answer can still miss the key facts, and an answer with all key facts can
still add invented claims."""

st.set_page_config(page_title="Golden dataset review", layout="wide")


@st.cache_resource(show_spinner="Reading the Reuters archive...")
def articles() -> dict[str, tuple[str, dict]]:
    download_archive()
    return {doc_id: (content, metadata) for doc_id, content, metadata in iter_articles()}


@st.cache_data(show_spinner=False)
def expected_distances(question: str, document_ids: tuple[str, ...]) -> dict[str, float]:
    """Live distance from the question to each expected article; one embedding call per question."""
    from app.services import vector  # imported here: needs OPENAI_API_KEY, only the distances use it

    return vector.document_distances(question, list(document_ids))


def load_runs(auto_filters: bool) -> list[tuple[str, dict[str, dict]]]:
    """All evaluation runs with this auto_filters setting, newest first, results keyed by case id.

    Auto-filter runs only count in the default topic mode (as /query runs); runs comparing other
    modes would mix in candidates the real system never retrieves. Older runs predate topic_mode.
    """
    from app.config import Settings  # here: get_settings() would need OPENAI_API_KEY

    default_mode = Settings.model_fields["topic_mode"].default
    runs = []
    for path in sorted(RUNS_DIR.glob("*-retrieval.json"), reverse=True):
        run = json.loads(path.read_text())
        mode = run["config"].get("topic_mode", "filter")
        if run["config"]["auto_filters"] == auto_filters and (not auto_filters or mode == default_mode):
            runs.append((path.name, {r["id"]: r for r in run["results"]}))
    return runs


def load_answer_runs() -> list[tuple[str, dict[str, dict]]]:
    """All answer-scoring runs (evals/score_answers.py), newest first, results keyed by case id."""
    return [
        (path.name, {r["id"]: r for r in json.loads(path.read_text())["results"]})
        for path in sorted(RUNS_DIR.glob("*-answers.json"), reverse=True)
    ]


def answer_issues(result: dict) -> list[str]:
    """What to check in one scored answer; empty if both fact checks agree and it is grounded."""
    issues = []
    for f in result["facts"]:
        if f["stated"] and not f["verbatim"]:
            issues.append(f"paraphrase? {f['fact']!r}")
        elif f["verbatim"] and f["stated"] is False:
            issues.append(f"judge disagrees on {f['fact']!r}")
        elif not f["verbatim"] and f["stated"] is False:
            issues.append(f"missing {f['fact']!r}")
        if f["stated"] and not f["quote_found"]:
            issues.append(f"judge quote not in answer for {f['fact']!r}")
    if result["grounded"] == "no":
        issues.append("not grounded")
    issues += [f"{name} failed" for name in result.get("errors", {})]
    return issues


def answer_label(run_name: str, result: dict) -> str:
    stamp = datetime.strptime(run_name[:15], "%Y%m%d-%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    issues = answer_issues(result)
    return f"{stamp} · " + (f"⚠️ {'; '.join(issues)}" if issues else "✅ facts stated, grounded")


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


def run_label(run_name: str, result: dict) -> str:
    """'2026-10-19 14:18:42 · ⚠️ filtered out (acq)' for a run file name and one case's result in it."""
    stamp = datetime.strptime(run_name[:15], "%Y%m%d-%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    if result["rank"] == 1:
        outcome = "✅ rank 1"
    elif result["filtered_out"]:
        outcome = f"⚠️ filtered out ({result['applied_filters'].get('topic') or 'dates'})"
    else:
        outcome = f"⚠️ rank {result['rank']}" if result["rank"] else "⚠️ not retrieved"
    return f"{stamp} · {outcome}"


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


JUDGE_MARK = '<mark style="background-color: #a5d8ff">'


def highlight(text: str, facts: list[str], quotes: list[str] = (), monospace: bool = True) -> str:
    """Text as HTML with each key fact marked yellow and each judge quote blue, ignoring case and
    line breaks. A key fact wins where the two overlap: blue marks only what the verbatim check missed.
    """
    kinds = [None] * len(text)  # per character: "fact", "quote" or None
    for phrases, kind in ((quotes, "quote"), (facts, "fact")):  # facts last, so they win
        for phrase in phrases:
            if tokens := phrase.split():
                pattern = r"\s+".join(re.escape(token) for token in tokens)
                for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                    kinds[m.start() : m.end()] = [kind] * (m.end() - m.start())
    parts, start = [], 0
    for i in range(1, len(text) + 1):
        if i == len(text) or kinds[i] != kinds[start]:
            segment = html.escape(text[start:i])
            opening = {"fact": "<mark>", "quote": JUDGE_MARK}.get(kinds[start])
            parts.append(f"{opening}{segment}</mark>" if opening else segment)
            start = i
    font = "font-family: ui-monospace, monospace; font-size: 0.85rem; " if monospace else ""
    return f'<div style="white-space: pre-wrap; {font}line-height: 1.45">{"".join(parts)}</div>'


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
        message = f"Not saved: unknown article ids {unknown}" if unknown else "Not saved: no expected article"
        state.flash = ("error", message)
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

    saved = f"Saved {cid}" + (" as reviewed" if reviewed else "")
    texts = {doc_id: articles()[doc_id][0] for doc_id in expected}
    if problems := case_problems(case, texts):
        state.flash = ("warning", f"{saved}, but it breaks a rule: " + "; ".join(problems))
    else:
        state.flash = ("success", saved)
    if next_id:
        state.case_id = next_id


def add_expected(cid: str, doc_id: str) -> None:
    key = f"expected_{cid}"
    if doc_id not in lines(state[key]):
        state[key] = state[key].rstrip() + "\n" + doc_id


def facts_label(facts: list[str], text: str) -> str:
    """'key facts 2/2', or 'key facts 1/2 ✗' when the article lacks some."""
    if not facts:
        return "no key facts"
    n_found = len(facts) - len(missing_facts(facts, text))
    return f"key facts {n_found}/{len(facts)}{'' if n_found == len(facts) else ' ✗'}"


@st.cache_data(show_spinner="Looking for similar articles...")
def duplicate_candidates(expected: tuple[str, ...], facts: tuple[str, ...]) -> list[dict]:
    texts = {doc_id: content for doc_id, (content, _) in articles().items()}
    return duplicates.candidates({"expected_doc_ids": list(expected), "key_facts": list(facts)}, texts)


def duplicate_expander(cid: str, candidate: dict, facts: list[str], nested: bool = False) -> None:
    doc_id = candidate["document_id"]
    content, metadata = articles()[doc_id]
    header = (
        f"{doc_id} · {metadata.get('date', '?')} · distance {candidate['distance']:.3f} to "
        f"{candidate['similar_to']} · {facts_label(facts, content)} · {metadata['title']}"
    )
    # Streamlit does not allow expanders inside expanders; nested candidates use a plain block.
    container = st.container(border=True) if nested else st.expander(header)
    with container:
        if nested:
            st.markdown(f"**{header}**")
        st.markdown(highlight(content, facts), unsafe_allow_html=True)
        missing = candidate["missing"]
        st.button(
            "Add to expected",
            key=f"dup_{cid}_{doc_id}",
            on_click=add_expected,
            args=(cid, doc_id),
            help=(
                f"Lacks key facts {missing}: adding it will be flagged until the facts or the "
                "expected articles are adjusted."
                if missing
                else "Only if a user would be fully satisfied with this article alone. Click Save afterwards."
            ),
        )


# --- layout --------------------------------------------------------------------------------

cases = load_cases(GOLDEN)
runs = {label: load_runs(auto) for label, auto in (("with auto filters", True), ("without auto filters", False))}
answer_runs = load_answer_runs()


def is_miss(cid: str) -> bool:
    """Not ranked first in the latest run of either setting."""
    return any(r and cid in r[0][1] and r[0][1][cid]["rank"] != 1 for r in runs.values())


def has_answer_issue(cid: str) -> bool:
    """The latest answer run with this case flags it: checks disagree, a fact is missing, or not grounded.

    The latest run containing the case, not the latest run: a run with --ids scores only a few cases.
    """
    latest = next((run[cid] for _, run in answer_runs if cid in run), None)
    return latest is not None and bool(answer_issues(latest))


with st.sidebar:
    st.header("Golden dataset")
    n_reviewed = sum(bool(c.get("reviewed")) for c in cases)
    st.progress(n_reviewed / len(cases) if cases else 0.0, text=f"{n_reviewed} of {len(cases)} reviewed")
    view = st.radio("Show", ["Unreviewed", "Misses in latest runs", "Answers to check", "All"])
    if view == "Unreviewed" and n_reviewed == len(cases):
        # Nothing left to review: show everything rather than an empty page.
        st.success("All cases are reviewed. Showing all.")
        view = "All"
    shown = [
        c
        for c in cases
        if view == "All"
        or (view == "Unreviewed" and not c.get("reviewed"))
        or (view == "Misses in latest runs" and is_miss(c["id"]))
        or (view == "Answers to check" and has_answer_issue(c["id"]))
    ]
    if not shown:
        st.info("Nothing to show in this view. Choose 'All' to see every case.")
        st.stop()
    ids = [c["id"] for c in shown]
    if state.get("case_id") not in ids:
        state.case_id = ids[0]
    labels = {
        c["id"]: f"{'✅' if c.get('reviewed') else '⬜'} {c['id']}"
        f"{'  ⚠️ miss' if is_miss(c['id']) else ''}{'  🔎 answer' if has_answer_issue(c['id']) else ''}"
        for c in shown
    }
    st.selectbox("Case", ids, key="case_id", format_func=labels.get)
    st.caption("✅ reviewed · ⚠️ not ranked first in a latest run · 🔎 latest answer run flags it")
    for label, label_runs in runs.items():
        st.caption(f"Runs {label}: {len(label_runs)}, latest {label_runs[0][0] if label_runs else 'none'}")
    st.caption(f"Answer runs: {len(answer_runs)}, latest {answer_runs[0][0] if answer_runs else 'none'}")

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
        "article answer it? Is the reference answer correct and complete? Does the question style match?\n\n"
        "**Key facts** are what a correct answer must add beyond the question: no names or terms from the "
        "question, as short as possible (usually a number, a name or a few words), copied verbatim from the "
        "article (✓), and only facts a good answer cannot leave out.\n\n"
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
    st.radio(
        "Question style",
        STYLES,
        key=f"style_{cid}",
        horizontal=True,
        help="Describes the question, not the article (every article has a date).\n\n"
        "**dated**: the question mentions a date or period, such as 'on March 24', "
        "'in April' or 'in Jan/Feb 1987'.\n\n"
        "**undated**: the question mentions no date or period.\n\n"
        "Only a dated question gives filter extraction a date to work with, so scores are "
        "reported per style.",
    )
    st.text_area("Reference answer", key=f"answer_{cid}", height=110)
    st.text_area(
        "Key facts (one per line)",
        key=f"facts_{cid}",
        height=90,
        help="What a correct answer must add beyond the question. Not repeated from the question, "
        "as short as possible, verbatim from the article, essential. Key facts are the minimal "
        "core without which the answer is wrong or incomplete. ✓/✗ below checks 'not repeated' "
        "and 'verbatim' automatically; the rest is your judgment.",
    )

    expected_ids = [d for d in lines(state[f"expected_{cid}"]) if d in articles()]
    for fact in lines(state[f"facts_{cid}"]):
        lacking = [d for d in expected_ids if missing_facts([fact], articles()[d][0])]
        if content_words(fact) <= content_words(state[f"question_{cid}"]):
            st.caption(f"✗ {fact}: only repeats the question")
        elif lacking:
            st.caption(
                f"✗ {fact}: not verbatim in {', '.join(lacking)}. Shorten the fact, "
                "or remove the article if it does not give this fact"
            )
        else:
            st.caption(f"✓ {fact}")

    st.text_area("Expected article ids (one per line)", key=f"expected_{cid}", height=90)
    st.text_area("Notes (what you changed and why)", key=f"notes_{cid}", height=70)

    b1, b2, b3 = st.columns(3)
    b1.button("Save", on_click=save, args=(cid,), use_container_width=True)
    b2.button("✅ Reviewed, next", on_click=save, args=(cid, True, next_id), type="primary", use_container_width=True)
    b3.button("Mark unreviewed", on_click=save, args=(cid, False), use_container_width=True)
    st.caption(f"Case {position + 1} of {len(ids)} in this view. Saving writes to {GOLDEN}.")

with right:
    facts = lines(state[f"facts_{cid}"])
    known = tuple(d for d in lines(state[f"expected_{cid}"]) if d in articles())
    try:
        distances = expected_distances(state[f"question_{cid}"].strip(), known)
    except Exception as exc:  # the review works without distances, e.g. offline
        distances = {}
        st.caption(f"Distances unavailable: {exc}")
    for i, doc_id in enumerate(lines(state[f"expected_{cid}"])):
        if doc_id not in articles():
            st.warning(f"Unknown article id: {doc_id}")
            continue
        content, metadata = articles()[doc_id]
        distance = f" · distance {distances[doc_id]:.3f}" if doc_id in distances else ""
        header = (
            f"Expected: {doc_id} · {metadata.get('date', '?')} · "
            f"{', '.join(metadata.get('topics', [])) or 'no topics'}{distance} · {facts_label(facts, content)}"
        )
        with st.expander(header, expanded=i == 0):
            st.markdown(highlight(content, facts), unsafe_allow_html=True)

    st.markdown("**Possible duplicates**", help=DUPLICATES_HELP)
    try:
        found = duplicate_candidates(tuple(expected_ids), tuple(facts))
    except Exception as exc:  # the review works without suggestions, e.g. if the index is missing
        found = []
        st.caption(f"Duplicate search unavailable: {exc}")
    complete = [c for c in found if not c["missing"]]
    partial = [c for c in found if c["missing"]]
    if not found:
        st.caption(f"No other article within distance {MAX_DISTANCE} of an expected article.")
    for c in complete:
        duplicate_expander(cid, c, facts)
    if partial:
        with st.expander(f"{len(partial)} similar article(s) lacking key facts: likely a different report"):
            st.caption(
                "Same kind of story, other figures (e.g. last week's report). Add one only if it "
                "answers the question, and then shorten the key facts so it contains them."
            )
            for c in partial:
                duplicate_expander(cid, c, facts, nested=True)

st.divider()
st.markdown("#### Answer", help=ANSWER_HELP)
case_answers = [(name, run[cid]) for name, run in answer_runs if cid in run]
if not case_answers:
    st.caption("Not in any answer run. Run `python -m evals.score_answers` to score answers.")
else:
    selected = st.selectbox(
        "Answer run (newest first)",
        range(len(case_answers)),
        format_func=lambda i: answer_label(*case_answers[i]),
        key=f"answer_run_{cid}",
    )
    result = case_answers[selected][1]
    if result["question"] != state[f"question_{cid}"].strip():
        st.warning(f"This run used an earlier version of the question: {result['question']}")
    if result["key_facts"] != facts:
        st.warning(f"This run scored earlier key facts: {result['key_facts']}")
    retrieval_note = (
        f"expected article at rank {result['rank']}"
        if result["rank"]
        else "**no expected article retrieved**, so the answer could not contain the facts"
    )
    st.caption(f"Retrieval: {retrieval_note} · trace {result['trace_id']}")
    # Blue only where the verbatim check missed the fact: there the judge's quote shows what it accepted.
    quotes = [f["quote"] for f in result["facts"] if f["stated"] and not f["verbatim"] and f["quote"]]
    st.markdown(highlight(result["answer"], result["key_facts"], quotes, monospace=False), unsafe_allow_html=True)

    def mark(value: bool | None) -> str:
        return "–" if value is None else ("✓" if value else "✗")

    rows = ["| Key fact | Verbatim | Judge | Judge's quote | Judge's reason |", "|---|---|---|---|---|"]
    for f in result["facts"]:
        quote = f["quote"] + ("" if f["quote_found"] or not f["stated"] else " ⚠️ not in the answer")
        cells = [f["fact"], mark(f["verbatim"]), mark(f["stated"]), quote, f["reason"] or ""]
        rows.append("| " + " | ".join(c.replace("|", "\\|").replace("\n", " ") for c in cells) + " |")
    st.markdown("\n".join(rows))

    grounded = {"yes": "✓ grounded", "no": "✗ not grounded"}.get(result["grounded"], "– not judged")
    with st.expander(f"Groundedness: {grounded}", expanded=result["grounded"] == "no"):
        st.markdown(result["groundedness_rationale"] or "")
    for name, error in result.get("errors", {}).items():
        st.error(f"{name} failed: {error}")

st.divider()
st.markdown("#### Evaluation runs", help=ADD_HELP)
st.caption(
    "The articles retrieved instead are the candidates for equally good answers. "
    "Open one to read it, and add it if it answers the question as well (hover ? for the rule)."
)
run_columns = st.columns(len(runs), gap="large")
expected_now = set(lines(state[f"expected_{cid}"]))
for column, (label, label_runs) in zip(run_columns, runs.items(), strict=True):
    with column:
        st.markdown(f"**{label}**")
        case_runs = [(name, run[cid]) for name, run in label_runs if cid in run]
        if not case_runs:
            st.caption("Not in any run.")
            continue
        st.markdown(f"All runs: {across_runs([result for _, result in case_runs])}")

        labels_by_index = [run_label(name, result) for name, result in case_runs]
        selected = st.selectbox(
            "Run (newest first)",
            range(len(case_runs)),
            format_func=labels_by_index.__getitem__,
            key=f"run_{label}_{cid}",
        )
        result = case_runs[selected][1]
        if result["question"] != state[f"question_{cid}"].strip():
            st.warning(f"This run used an earlier version of the question: {result['question']}")
        if set(result["expected_doc_ids"]) != expected_now:
            st.caption(f"Ranked against the expected articles at the time: {', '.join(result['expected_doc_ids'])}")
        if result["filtered_out"]:
            st.caption("Filtered out: the applied filters excluded every expected article.")
        if result["applied_filters"]:
            for doc_id in result["expected_doc_ids"]:
                if doc_id in articles():
                    st.caption(f"Filters vs {doc_id}: {filter_check(result['applied_filters'], articles()[doc_id][1])}")
        # Closest chunk per article, in rank order (several chunks of one article count once).
        retrieved: dict[str, float] = {}
        for hit in result["retrieved"]:
            retrieved.setdefault(hit["document_id"], hit["distance"])
        for doc_id, distance in retrieved.items():
            if doc_id not in articles():
                continue
            content, metadata = articles()[doc_id]
            mark = "✓ expected · " if doc_id in expected_now else ""
            topics = ", ".join(metadata.get("topics", [])) or "no topics"
            with st.expander(
                f"{mark}{doc_id} · {metadata.get('date', '?')} · {topics} · distance {distance:.3f} · "
                f"{facts_label(facts, content)} · {metadata['title']}"
            ):
                st.markdown(highlight(content, facts), unsafe_allow_html=True)
                if doc_id not in expected_now:
                    st.button(
                        "Add to expected",
                        key=f"add_{label}_{cid}_{doc_id}",
                        on_click=add_expected,
                        args=(cid, doc_id),
                        help="Only if a user would be fully satisfied with this article alone. "
                        "Adds it to the field above; click Save to write it.",
                    )
