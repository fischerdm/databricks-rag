"""Rerun the free fact checks on answers already stored in evals/runs/, without any API call.

Usage:
    python -m evals.rescore_verbatim [RUN.json ...]

For each answers run (default: every *-answers.json in evals/runs/), applies the verbatim and the
normalized check from evals/scorers.py to the stored answers and compares both with the judge's
stored verdicts. A change to the free checks can be measured this way, on the very answers the
judge saw. Each result keeps the key facts it was scored with, so later case fixes do not mix in.
"""

import argparse
import json
from pathlib import Path

from evals.checks import missing_facts, missing_normalized
from evals.evaluate import RUNS_DIR


def rescore(run: dict) -> list[dict]:
    """One row per key fact: verbatim, normalized, and the judge's stored verdict."""
    rows = []
    for result in run["results"]:
        not_verbatim = set(missing_facts(result["key_facts"], result["answer"]))
        not_normalized = set(missing_normalized(result["key_facts"], result["answer"]))
        for f in result["facts"]:
            rows.append(
                {
                    "id": result["id"],
                    "fact": f["fact"],
                    "quote": f["quote"],
                    "verbatim": f["fact"] not in not_verbatim,
                    "normalized": f["fact"] not in not_normalized,
                    "stated": f["stated"],
                }
            )
    return rows


def all_facts(rows: list[dict], key: str) -> float:
    """Share of cases where every key fact passes the check."""
    by_case: dict[str, bool] = {}
    for r in rows:
        by_case[r["id"]] = by_case.get(r["id"], True) and bool(r[key])
    return sum(by_case.values()) / len(by_case)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path, help="answers runs (default: all in evals/runs/)")
    args = parser.parse_args()
    paths = args.runs or sorted(RUNS_DIR.glob("*-answers.json"))

    print(f"{'run':<30} {'n':>3}  {'all facts':>26}  {'per fact':>26}")
    print(f"{'':<30} {'':>3}  {'verbatim':>8} {'normal.':>8} {'judge':>8}  {'verbatim':>8} {'normal.':>8} {'judge':>8}")
    gained, gap, false_matches = [], [], []
    for path in paths:
        rows = rescore(json.loads(path.read_text()))
        n_cases = len({r["id"] for r in rows})
        cells = [f"{all_facts(rows, k):>8.0%}" for k in ("verbatim", "normalized", "stated")]
        cells += [f"{sum(bool(r[k]) for r in rows) / len(rows):>8.0%}" for k in ("verbatim", "normalized", "stated")]
        print(f"{path.stem:<30} {n_cases:>3}  {' '.join(cells[:3])}  {' '.join(cells[3:])}")
        gained += [(path.stem, r) for r in rows if r["normalized"] and not r["verbatim"]]
        gap += [(path.stem, r) for r in rows if r["stated"] and not r["normalized"]]
        false_matches += [(path.stem, r) for r in rows if r["normalized"] and r["stated"] is False]

    for title, pairs in (
        ("Matched only after normalising", gained),
        ("Judge: stated, still not matched (paraphrase or number format?)", gap),
        ("Matched, but judge: not stated (normalising too loose, or judge wrong?)", false_matches),
    ):
        print(f"\n{title} ({len(pairs)}):")
        for run, r in pairs:
            print(f"  {run} {r['id']} {r['fact']!r}: judge quote {r['quote']!r}")


if __name__ == "__main__":
    main()
