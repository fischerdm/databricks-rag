"""Suggest likely duplicates of each case's expected articles, as candidates for expected_doc_ids.

Usage:
    python -m evals.duplicates [--golden evals/golden.jsonl] [--max-distance 0.15]

Reuters often sent one story several times: a repeat, an update, a short highlights item next
to a detailed table. Such versions have almost the same first chunk, so they are found by
embedding distance between first chunks. Distance alone is not enough: recurring reports (weekly
purchases, earnings tables) look alike but carry other figures. A candidate is therefore only
suggested if it contains every key fact of the case; the others are counted, not listed.

Nothing is changed: adding a candidate is a review decision.
"""

import argparse
from pathlib import Path

from app.services import vector
from evals.checks import missing_facts

# Known duplicates were at most 0.115 apart, related but different stories at least 0.197.
MAX_DISTANCE = 0.15


def candidates(case: dict, texts: dict[str, str], max_distance: float = MAX_DISTANCE) -> list[dict]:
    """Articles close to an expected article, with the key facts each one lacks.

    Sorted with the complete candidates (lacking nothing) first, then by distance.
    """
    expected = set(case["expected_doc_ids"])
    found: dict[str, dict] = {}
    for doc_id in case["expected_doc_ids"]:
        for other, distance in vector.similar_documents(doc_id, max_distance).items():
            if other in expected or other not in texts:
                continue
            if other not in found or distance < found[other]["distance"]:
                found[other] = {
                    "document_id": other,
                    "similar_to": doc_id,
                    "distance": distance,
                    "missing": missing_facts(case["key_facts"], texts[other]),
                }
    return sorted(found.values(), key=lambda c: (bool(c["missing"]), c["distance"]))


def main() -> None:
    from evals.generate import DEFAULT_OUT, load_cases
    from scripts.ingest_reuters import download_archive, iter_articles

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", type=Path, default=DEFAULT_OUT, help="golden dataset (JSONL)")
    parser.add_argument("--max-distance", type=float, default=MAX_DISTANCE, help="largest first-chunk distance")
    args = parser.parse_args()

    download_archive()
    articles = {doc_id: (content, metadata) for doc_id, content, metadata in iter_articles()}
    texts = {doc_id: content for doc_id, (content, _) in articles.items()}

    n_suggested = n_hidden = 0
    for case in load_cases(args.golden):
        found = candidates(case, texts, args.max_distance)
        complete = [c for c in found if not c["missing"]]
        n_hidden += len(found) - len(complete)
        if not complete:
            continue
        print(f"{case['id']}: {case['question']}")
        for c in complete:
            metadata = articles[c["document_id"]][1]
            print(f"  + {c['document_id']} ({metadata.get('date', '?')}, distance {c['distance']:.3f} "
                  f"to {c['similar_to']}): {metadata['title']}")
        n_suggested += len(complete)
    print(f"\n{n_suggested} candidates contain all key facts and are worth a look in the review UI. "
          f"{n_hidden} similar articles lack key facts and are not listed.")


if __name__ == "__main__":
    main()
