"""Run the golden-dataset evaluation against Vector Search and gate on thresholds (Databricks job).

Usage (as a bundle job, see databricks.yml):
    databricks bundle run evaluate [--params limit=5]

Runs the same scripts as locally, evals.evaluate (retrieval) and evals.score_answers (answers,
scorers and judges), with VECTOR_BACKEND=databricks. Both log to the workspace's MLflow experiment:
score_answers its run with every question as a trace, this script a run with the retrieval metrics.
The job fails if a result is below its threshold, which makes it usable as a CI gate.

The scripts run as subprocesses in a temporary copy of the repo: serverless runs this file inside
a kernel with its own event loop, which the scripts' asyncio.run() does not tolerate, and their
evals/runs/ output stays out of the bundle's files.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.runtime import dbutils

AUTH_VARIABLES = {
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "DATABRICKS_CLIENT_ID",
    "DATABRICKS_CLIENT_SECRET",
    "DATABRICKS_AUTH_TYPE",
    "DATABRICKS_CONFIG_PROFILE",
    "DATABRICKS_USERNAME",
    "DATABRICKS_PASSWORD",
}


def run_script(module: str, args: list[str], cwd: Path, env: dict, kind: str) -> dict:
    """Run `python -m module`, print its output and return the run record it wrote to evals/runs/."""
    result = subprocess.run([sys.executable, "-m", module, *args], cwd=cwd, env=env, capture_output=True, text=True)
    # The subprocess writes to the real stdout, which the job output does not show; print it here.
    print(result.stdout, result.stderr[-5000:], sep="\n", flush=True)
    result.check_returncode()
    return json.loads(max((cwd / "evals" / "runs").glob(f"*-{kind}.json")).read_text())


def script_env(args: argparse.Namespace) -> dict:
    """Environment for the scripts: the app's settings, the API keys and Databricks credentials."""
    w = WorkspaceClient()
    token = w.config.authenticate()["Authorization"].removeprefix("Bearer ")
    # Drop other credentials the runtime may set, so the SDK and MLflow in the subprocess see exactly
    # one way to authenticate: the host and token below. Other DATABRICKS_* variables stay; MLflow
    # needs DATABRICKS_RUNTIME_VERSION, for instance.
    env = {k: v for k, v in os.environ.items() if k not in AUTH_VARIABLES}
    return env | {
        "DATABRICKS_HOST": w.config.host,
        "DATABRICKS_TOKEN": token,
        "OPENAI_API_KEY": dbutils.secrets.get(args.secret_scope, "openai_api_key"),
        "ANTHROPIC_API_KEY": dbutils.secrets.get(args.secret_scope, "anthropic_api_key"),
        "VECTOR_BACKEND": "databricks",
        "DATABRICKS_INDEX": args.index,
        "MLFLOW_TRACKING_URI": "databricks",
        "MLFLOW_EXPERIMENT_NAME": args.experiment,
    }


def gate(retrieval: dict, answers: dict, args: argparse.Namespace) -> list[str]:
    """Print each gated metric against its threshold; return the names of those below it."""
    checks = {
        "hit@k": (retrieval["summary"].get("hit@k"), args.min_hit_at_k),
        "all_facts_judged": (answers["summary"].get("all_facts_judged"), args.min_all_facts),
        "grounded": (answers["summary"].get("grounded"), args.min_grounded),
    }
    print(f"\n{'metric':<18} {'value':>7} {'min':>7}")
    failed = []
    for name, (value, minimum) in checks.items():
        ok = value is not None and value >= minimum
        shown = "–" if value is None else f"{value:.0%}"
        print(f"{name:<18} {shown:>7} {minimum:>7.0%}  {'ok' if ok else 'BELOW'}")
        if not ok:
            failed.append(name)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", required=True, help="where the bundle synced the repo")
    parser.add_argument("--index", required=True, help="Vector Search index (catalog.schema.name)")
    parser.add_argument("--experiment", required=True, help="MLflow experiment path")
    parser.add_argument("--secret-scope", default="rag")
    parser.add_argument("--limit", type=int, default=0, help="answer at most N cases (0: all); retrieval uses all")
    # Somewhat below the baseline (docs/evaluation.md), since LLM steps vary by a few points between runs.
    parser.add_argument("--min-hit-at-k", type=float, default=0.90)
    parser.add_argument("--min-all-facts", type=float, default=0.90)
    parser.add_argument("--min-grounded", type=float, default=0.95)
    args = parser.parse_args()

    env = script_env(args)
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        shutil.copytree(args.repo_root, repo, ignore=shutil.ignore_patterns(".databricks", "__pycache__"))
        retrieval = run_script("evals.evaluate", ["--reviewed-only"], repo, env, "retrieval")
        answer_args = ["--reviewed-only"] + (["--limit", str(args.limit)] if args.limit else [])
        answers = run_script("evals.score_answers", answer_args, repo, env, "answers")

    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(args.experiment)
    config = retrieval["config"]
    run_name = f"retrieval top_k={config['top_k']} auto_filters={config['auto_filters']}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config)
        mlflow.log_metrics({f"summary/{k}": v for k, v in retrieval["summary"].items() if v is not None})
        mlflow.log_dict(retrieval, "retrieval.json")
    # The answer run's full record (verdicts, quotes) next to its traces, e.g. for the review app.
    with mlflow.start_run(run_id=answers["config"]["mlflow_run_id"]):
        mlflow.log_dict(answers, "answers.json")

    if answers["summary"].get("errors"):
        print(f"\nWarning: scorer errors in {answers['summary']['errors']} cases (see the output above)")
    if failed := gate(retrieval, answers, args):
        raise SystemExit(f"Evaluation below threshold: {', '.join(failed)}")
    print("\nEvaluation passed.")


if __name__ == "__main__":
    main()
