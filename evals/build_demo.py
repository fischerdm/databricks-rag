"""Build the snapshot behind the read-only review app on Streamlit Community Cloud.

Usage (locally, with the index built and OPENAI_API_KEY set):
    python -m evals.build_demo

The review app needs the vector index and an OpenAI key for two things: the distance from each
question to its expected articles, and the duplicate candidates. It also shows the evaluation runs,
which are git-ignored. The deployed app has none of these, so this script computes the first two
for every case in evals/golden.jsonl and copies the runs the app shows into evals/demo/.
With REVIEW_DEMO=1, the app reads from there and disables editing.

Rerun it after changing the golden dataset or adding runs, and commit evals/demo/.
"""

import json
import shutil
from datetime import date
from pathlib import Path

from app.config import Settings
from app.services import vector
from evals import duplicates
from evals.checks import DEFAULT_OUT, load_cases
from evals.evaluate import RUNS_DIR
from scripts.ingest_reuters import download_archive, iter_articles

DEMO_DIR = Path("evals/demo")


def shown_by_app(path: Path) -> bool:
    """The runs the review app lists: answer runs, and retrieval runs as /query runs or without filters.

    Mirrors load_runs in evals/review_app.py; experiments with other topic modes are left out.
    """
    if path.name.endswith("-answers.json"):
        return True
    config = json.loads(path.read_text())["config"]
    default_mode = Settings.model_fields["topic_mode"].default
    return not config["auto_filters"] or config.get("topic_mode", "filter") == default_mode


def main() -> None:
    download_archive()
    texts = {doc_id: content for doc_id, content, _ in iter_articles()}
    cases = load_cases(DEFAULT_OUT)
    data = {"built": date.today().isoformat(), "distances": {}, "duplicates": {}}
    for case in cases:
        known = [d for d in case["expected_doc_ids"] if d in texts]
        data["distances"][case["id"]] = vector.document_distances(case["question"], known)
        data["duplicates"][case["id"]] = duplicates.candidates(case, texts)
        print(f"{case['id']}: {len(data['duplicates'][case['id']])} duplicate candidates")

    runs_dir = DEMO_DIR / "runs"
    shutil.rmtree(runs_dir, ignore_errors=True)
    runs_dir.mkdir(parents=True)
    runs = [p for p in sorted(RUNS_DIR.glob("*.json")) if shown_by_app(p)]
    for path in runs:
        shutil.copy2(path, runs_dir / path.name)
    (DEMO_DIR / "data.json").write_text(json.dumps(data, indent=1) + "\n")
    print(f"\n{len(cases)} cases and {len(runs)} runs written to {DEMO_DIR}/. Commit it to update the deployed app.")


if __name__ == "__main__":
    main()
