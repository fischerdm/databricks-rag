"""Score retrieval against the golden dataset.

Usage:
    python -m evals.evaluate [--golden evals/golden.jsonl] [--top-k N] [--no-auto-filters] [--reviewed-only]

Runs every question through the same retrieval pipeline as /query (filter extraction, filtered
search, fallback) and checks where the expected articles rank. Prints a summary and the misses,
and writes the full results to evals/runs/ so runs can be compared after a change.

Metrics, over the cases whose expected articles are indexed:
- hit@1, hit@k: share of cases where an expected article is the first / among the retrieved ones.
- MRR: mean of 1/rank of the first expected article (0 if not retrieved).
- filtered out: share of cases where the applied filters excluded every expected article, so
  search could not find it at any rank. Points at filter extraction rather than the embeddings.
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from app.config import get_settings
from app.schemas import MetadataValue, RetrievedChunk, SearchFilters
from app.services import retrieval, vector
from evals.generate import DEFAULT_OUT, load_cases

RUNS_DIR = Path("evals/runs")


def doc_rank(chunks: list[RetrievedChunk], expected: set[str]) -> int | None:
    """1-based rank of the first expected article among the retrieved articles, or None.

    Ranks articles, not chunks: several chunks of one article count as one position.
    """
    doc_ids = list(dict.fromkeys(chunk.metadata.get("document_id") for chunk in chunks))
    for rank, doc_id in enumerate(doc_ids, start=1):
        if doc_id in expected:
            return rank
    return None


def admits(filters: SearchFilters, metadata: dict[str, MetadataValue]) -> bool:
    """Whether a document with this metadata passes the filters (mirrors vector.build_filter)."""
    date_int = metadata.get("date_int")
    if filters.date_from and (date_int is None or date_int < int(filters.date_from.strftime("%Y%m%d"))):
        return False
    if filters.date_to and (date_int is None or date_int > int(filters.date_to.strftime("%Y%m%d"))):
        return False
    if filters.topic and filters.topic.lower() not in metadata.get("topics", []):
        return False
    return True


async def evaluate_case(case: dict, top_k: int, auto_filters: bool, metadata: dict) -> dict:
    expected = {doc_id for doc_id in case["expected_doc_ids"] if doc_id in metadata}
    applied, chunks = await retrieval.retrieve(case["question"], top_k, SearchFilters(), auto_filters)
    rank = doc_rank(chunks, expected)
    return {
        "id": case["id"],
        "question": case["question"],
        "style": case.get("style"),
        "expected_doc_ids": sorted(expected),
        "rank": rank,
        "filtered_out": not any(admits(applied, metadata[doc_id]) for doc_id in expected),
        "applied_filters": applied.model_dump(mode="json", exclude_none=True),
        "retrieved": [
            {"document_id": c.metadata.get("document_id"), "title": c.metadata.get("title"), "distance": c.distance}
            for c in chunks
        ],
    }


def summarize(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "hit@1": sum(r["rank"] == 1 for r in results) / n,
        "hit@k": sum(r["rank"] is not None for r in results) / n,
        "mrr": sum(1 / r["rank"] for r in results if r["rank"]) / n,
        "filtered_out": sum(r["filtered_out"] for r in results) / n,
    }


def print_report(summary: dict, by_style: dict[str, dict], results: list[dict], top_k: int) -> None:
    def row(label: str, s: dict) -> str:
        if not s["n"]:
            return f"{label:<10} {0:>4}"
        return (
            f"{label:<10} {s['n']:>4} {s['hit@1']:>7.0%} {s['hit@k']:>7.0%} "
            f"{s['mrr']:>6.2f} {s['filtered_out']:>9.0%}"
        )

    print(f"\n{'':<10} {'n':>4} {'hit@1':>7} {f'hit@{top_k}':>7} {'MRR':>6} {'filt.out':>9}")
    print(row("all", summary))
    for style, s in by_style.items():
        print(row(style, s))

    misses = [r for r in results if r["rank"] != 1]
    if misses:
        print(f"\nNot ranked first ({len(misses)}):")
    for r in misses:
        rank = f"rank {r['rank']}" if r["rank"] else "not retrieved"
        reason = ", filtered out" if r["filtered_out"] else ""
        print(f"\n  {r['id']} [{rank}{reason}] {r['question']}")
        print(f"    expected: {', '.join(r['expected_doc_ids'])}   filters: {r['applied_filters'] or 'none'}")
        for hit in r["retrieved"]:
            print(f"    - {hit['document_id']}: {hit['title']}")


async def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", type=Path, default=DEFAULT_OUT, help="golden dataset (JSONL)")
    parser.add_argument("--top-k", type=int, default=settings.top_k, help="chunks retrieved per question")
    parser.add_argument("--no-auto-filters", action="store_true", help="search without filter extraction")
    parser.add_argument("--reviewed-only", action="store_true", help="skip cases not marked as reviewed")
    args = parser.parse_args()
    auto_filters = not args.no_auto_filters

    cases = load_cases(args.golden)
    if args.reviewed_only:
        cases = [c for c in cases if c.get("reviewed")]
    metadata = vector.document_metadata(sorted({d for c in cases for d in c["expected_doc_ids"]}))
    # A case whose articles are not indexed cannot be scored; count it separately instead of as a miss.
    scorable = [c for c in cases if any(d in metadata for d in c["expected_doc_ids"])]

    semaphore = asyncio.Semaphore(8)

    async def limited(case):
        async with semaphore:
            return await evaluate_case(case, args.top_k, auto_filters, metadata)

    results = await asyncio.gather(*(limited(c) for c in scorable))
    summary = summarize(results)
    by_style = {
        style: summarize([r for r in results if r["style"] == style])
        for style in sorted({r["style"] for r in results if r["style"]})
    }

    print(
        f"{len(cases)} cases ({sum(bool(c.get('reviewed')) for c in cases)} reviewed), "
        f"{len(cases) - len(scorable)} skipped (articles not indexed). "
        f"top_k={args.top_k}, auto_filters={auto_filters}"
    )
    print_report(summary, by_style, results, args.top_k)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_path = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-retrieval.json"
    run = {
        "config": {
            "golden": str(args.golden),
            "top_k": args.top_k,
            "auto_filters": auto_filters,
            "reviewed_only": args.reviewed_only,
            "filter_model": settings.filter_model,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
        "summary": summary,
        "by_style": by_style,
        "results": results,
    }
    run_path.write_text(json.dumps(run, indent=2))
    print(f"\nFull results: {run_path}")


if __name__ == "__main__":
    asyncio.run(main())
