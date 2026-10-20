"""Score answers against the golden dataset with MLflow.

Usage:
    python -m evals.score_answers [--golden evals/golden.jsonl] [--top-k N] [--no-auto-filters]
                                  [--reviewed-only] [--limit N] [--ids q0001 q0002 ...]

Runs every question through retrieval and answer generation as /query does, records each as an
MLflow trace (retrieval as a RETRIEVER span), then scores the traces with mlflow.genai.evaluate
and the scorers in evals/scorers.py. Prints a summary and every disagreement between the verbatim
check and the judge, and writes the full results to evals/runs/ for the review app.

The answers are generated here rather than by evaluate's predict_fn: it runs each call in a new
event loop, and the app's async OpenAI client is bound to one loop.

Results are logged to the MLflow tracking store in MLFLOW_TRACKING_URI (default: sqlite:///mlflow.db).
View them with:
    .venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
import asyncio
import json
import os
from datetime import datetime

import mlflow
from mlflow.entities import Document, SpanType

from app.config import get_settings
from app.schemas import SearchFilters
from app.services import llm, retrieval, vector
from evals import scorers
from evals.evaluate import RUNS_DIR, doc_rank
from evals.generate import DEFAULT_OUT, load_cases

EXPERIMENT = "rag-answers"


async def answer_case(case: dict, top_k: int, auto_filters: bool, expected: set[str]) -> dict:
    """Retrieve and answer one question inside an MLflow trace; return what the report needs."""
    with mlflow.start_span(name="rag", span_type=SpanType.CHAIN) as root:
        root.set_inputs({"question": case["question"]})
        mlflow.update_current_trace(tags={"case_id": case["id"]})
        with mlflow.start_span(name="retrieve", span_type=SpanType.RETRIEVER) as span:
            span.set_inputs({"question": case["question"], "top_k": top_k})
            applied, _, chunks = await retrieval.retrieve(case["question"], top_k, SearchFilters(), auto_filters)
            span.set_attributes({"applied_filters": applied.model_dump(mode="json", exclude_none=True)})
            span.set_outputs(
                [Document(page_content=c.text, metadata={"doc_uri": c.metadata.get("document_id")}) for c in chunks]
            )
        answer = await llm.generate_answer(case["question"], chunks)
        root.set_outputs(answer)
    return {
        "id": case["id"],
        "question": case["question"],
        "style": case.get("style"),
        "key_facts": case["key_facts"],
        "expected_doc_ids": sorted(expected),
        # An answer can only contain the facts if an expected article was retrieved.
        "rank": doc_rank(chunks, expected),
        "applied_filters": applied.model_dump(mode="json", exclude_none=True),
        "retrieved": [{"document_id": c.metadata.get("document_id"), "distance": c.distance} for c in chunks],
        "answer": answer,
        "trace_id": root.trace_id,
    }


def assessment(row, name: str) -> dict:
    """One scorer's result for one row of evaluate's result_df: value, rationale, metadata, error."""
    found = next((a for a in row["assessments"] if a["assessment_name"] == name), None)
    if found is None:
        return {"value": None, "rationale": None, "metadata": {}, "error": "no assessment"}
    feedback = found.get("feedback") or {}
    error = feedback.get("error")
    value = feedback.get("value")
    return {
        "value": value,
        "rationale": found.get("rationale"),
        "metadata": found.get("metadata") or {},
        "error": (error.get("error_message") or str(error)) if error else None,
    }


def collect(result_df, answered: list[dict]) -> list[dict]:
    """Merge the scorers' results into the per-case records, keyed by trace id."""
    by_trace = {row["trace_id"]: row for _, row in result_df.iterrows()}
    results = []
    for case in answered:
        row = by_trace[case["trace_id"]]
        verbatim = assessment(row, "key_facts_verbatim")
        coverage = assessment(row, "fact_coverage")
        grounded = assessment(row, "groundedness")
        missing = set(json.loads(verbatim["metadata"].get("missing", "[]")))
        verdicts = json.loads(coverage["metadata"].get("verdicts", "[]"))
        judged = {v["fact"]: v for v in verdicts}
        facts = [
            {
                "fact": fact,
                "verbatim": fact not in missing,
                "stated": judged[fact]["stated"] if fact in judged else None,
                "quote": judged[fact]["quote"] if fact in judged else "",
                "quote_found": judged[fact]["quote_found"] if fact in judged else False,
                "reason": judged[fact]["reason"] if fact in judged else coverage["error"],
            }
            for fact in case["key_facts"]
        ]
        results.append(
            {
                **case,
                "facts": facts,
                "all_verbatim": verbatim["value"],
                "all_stated": coverage["value"] if not coverage["error"] else None,
                "grounded": grounded["value"] if not grounded["error"] else None,
                "groundedness_rationale": grounded["rationale"],
                "errors": {
                    name: a["error"]
                    for name, a in (("fact_coverage", coverage), ("groundedness", grounded))
                    if a["error"]
                },
            }
        )
    return results


def share(values: list) -> float | None:
    known = [v for v in values if v is not None]
    return sum(known) / len(known) if known else None


def summarize(results: list[dict]) -> dict:
    facts = [f for r in results for f in r["facts"]]
    return {
        "n": len(results),
        "retrieved": share([r["rank"] is not None for r in results]),
        "all_facts_verbatim": share([r["all_verbatim"] for r in results]),
        "all_facts_judged": share([r["all_stated"] for r in results]),
        "grounded": share([None if r["grounded"] is None else r["grounded"] == "yes" for r in results]),
        "facts": len(facts),
        "facts_verbatim": share([f["verbatim"] for f in facts]),
        "facts_judged": share([f["stated"] for f in facts]),
        "errors": sum(bool(r["errors"]) for r in results),
    }


def print_report(summary: dict, by_retrieval: dict[str, dict], results: list[dict]) -> None:
    def pct(x: float | None) -> str:
        return "   –" if x is None else f"{x:>4.0%}"

    print(f"\n{'':<22} {'n':>3}  {'all facts':>17}  {'per fact':>17}  {'grounded':>8}")
    print(f"{'':<22} {'':>3}  {'verbatim':>8} {'judge':>8}  {'verbatim':>8} {'judge':>8}")
    for label, s in {"all": summary, **by_retrieval}.items():
        if s["n"]:
            print(
                f"{label:<22} {s['n']:>3}  {pct(s['all_facts_verbatim']):>8} {pct(s['all_facts_judged']):>8}  "
                f"{pct(s['facts_verbatim']):>8} {pct(s['facts_judged']):>8}  {pct(s['grounded']):>8}"
            )
    if summary["errors"]:
        print(f"\nScorer errors in {summary['errors']} cases:")
        for r in results:
            for name, error in r["errors"].items():
                print(f"  {r['id']} {name}: {error}")

    # Where the two fact checks disagree, one of them is wrong: these are the cases to read.
    paraphrased = [(r, f) for r in results for f in r["facts"] if f["stated"] and not f["verbatim"]]
    contradicted = [(r, f) for r in results for f in r["facts"] if f["verbatim"] and f["stated"] is False]
    unverified = [(r, f) for r in results for f in r["facts"] if f["stated"] and not f["quote_found"]]
    for title, pairs in (
        ("Judge: stated, but not verbatim (paraphrase?)", paraphrased),
        ("Verbatim, but judge: not stated (fact too loose, or judge wrong?)", contradicted),
        ("Judge's quote not found in the answer", unverified),
    ):
        if pairs:
            print(f"\n{title} ({len(pairs)}):")
        for r, f in pairs:
            print(f"  {r['id']} {f['fact']!r}: quote {f['quote']!r}. {f['reason']}")

    ungrounded = [r for r in results if r["grounded"] == "no"]
    if ungrounded:
        print(f"\nNot grounded in the retrieved passages ({len(ungrounded)}):")
    for r in ungrounded:
        print(f"\n  {r['id']} {r['question']}\n    answer: {r['answer']}\n    judge: {r['groundedness_rationale']}")


async def answer_all(cases: list[dict], top_k: int, auto_filters: bool, metadata: dict) -> list[dict]:
    semaphore = asyncio.Semaphore(8)

    async def limited(case):
        async with semaphore:
            expected = {d for d in case["expected_doc_ids"] if d in metadata}
            return await answer_case(case, top_k, auto_filters, expected)

    return await asyncio.gather(*(limited(c) for c in cases))


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", default=DEFAULT_OUT, help="golden dataset (JSONL)")
    parser.add_argument("--top-k", type=int, default=settings.top_k, help="chunks retrieved per question")
    parser.add_argument("--no-auto-filters", action="store_true", help="search without filter extraction")
    parser.add_argument("--reviewed-only", action="store_true", help="skip cases not marked as reviewed")
    parser.add_argument("--limit", type=int, help="only the first N cases, for a cheap trial run")
    parser.add_argument("--ids", nargs="+", help="only these case ids, e.g. to rescore cases after a fix")
    args = parser.parse_args()
    auto_filters = not args.no_auto_filters
    if not settings.anthropic_api_key:
        parser.error("ANTHROPIC_API_KEY is not set (in .env or the environment); the judges need it")
    # MLflow's built-in judge reads the key from the environment, not from our settings.
    os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)

    cases = load_cases(args.golden)
    if args.reviewed_only:
        cases = [c for c in cases if c.get("reviewed")]
    if args.ids:
        if unknown := set(args.ids) - {c["id"] for c in cases}:
            parser.error(f"unknown case ids: {sorted(unknown)}")
        cases = [c for c in cases if c["id"] in args.ids]
    metadata = vector.document_metadata(sorted({d for c in cases for d in c["expected_doc_ids"]}))
    scorable = [c for c in cases if any(d in metadata for d in c["expected_doc_ids"])][: args.limit]

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT)
    # Adds every OpenAI call (filter extraction, embedding, answer) to the trace, with prompts and tokens.
    # silent: logging the filter extractor's parsed output triggers harmless Pydantic serializer warnings.
    mlflow.openai.autolog(silent=True)
    config = {
        "golden": str(args.golden),
        "n_cases": len(scorable),
        "top_k": args.top_k,
        "auto_filters": auto_filters,
        "topic_mode": settings.topic_mode,
        "reviewed_only": args.reviewed_only,
        "ids": ",".join(args.ids) if args.ids else "all",
        "chat_model": settings.chat_model,
        "judge_model": settings.judge_model,
        "filter_model": settings.filter_model,
        "embedding_model": settings.embedding_model,
    }
    with mlflow.start_run(run_name=f"answers top_k={args.top_k} auto_filters={auto_filters}") as run:
        mlflow.log_params(config)
        answered = asyncio.run(answer_all(scorable, args.top_k, auto_filters, metadata))
        mlflow.flush_trace_async_logging()
        data = [
            {"trace": mlflow.get_trace(a["trace_id"]), "expectations": {"key_facts": a["key_facts"]}} for a in answered
        ]
        evaluation = mlflow.genai.evaluate(
            data=data, scorers=[scorers.key_facts_verbatim, scorers.fact_coverage, scorers.groundedness()]
        )
        results = collect(evaluation.result_df, answered)
        summary = summarize(results)
        by_retrieval = {
            "expected retrieved": summarize([r for r in results if r["rank"] is not None]),
            "expected not retrieved": summarize([r for r in results if r["rank"] is None]),
        }
        mlflow.log_metrics({f"summary/{k}": v for k, v in summary.items() if v is not None})

    print(
        f"\n{len(cases)} cases, {len(scorable)} scored. top_k={args.top_k}, auto_filters={auto_filters}, "
        f"answers by {settings.chat_model}, judged by {settings.judge_model}"
    )
    print_report(summary, by_retrieval, results)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_path = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-answers.json"
    run_record = {
        "config": {**config, "mlflow_run_id": run.info.run_id},
        "summary": summary,
        "by_retrieval": by_retrieval,
        "results": results,
    }
    run_path.write_text(json.dumps(run_record, indent=2))
    print(f"\nFull results: {run_path}\nMLflow run: {run.info.run_id} (experiment '{EXPERIMENT}')")


if __name__ == "__main__":
    main()
