# From evaluation to CI on Databricks

How the golden dataset's fields relate to what the user sees, and how the evaluation would run as a CI gate on Databricks. It builds on [evaluation.md](evaluation.md), which explains the dataset, the metrics and the findings.

**Contents**

1. [What is compared with what](#1-what-is-compared-with-what)
2. [The evaluation as a CI gate on Databricks](#2-the-evaluation-as-a-ci-gate-on-databricks)

---

## 1. What is compared with what

The golden dataset is written in the articles' wording: Reuters' telex style, with 'mln', 'dlrs', 'pct'. The user sees the answer model's wording: 'million', 'dollars', '%'. That is fine, because the user's answer is never compared word for word with the reference answer.

| Field | Example (q0043) | Who reads it | Used in scoring? |
|---|---|---|---|
| `reference_answer` | 'A Taiwan mission plans to buy 410,000 bales of U.S. cotton worth about 110 mln U.S. dlrs.' | A reviewer, in the review app: what does a correct answer look like? | **No** |
| `key_facts` | '410,000 bales' | The fact checks | **Yes** |
| The app's answer | 'Taiwan plans to buy 410,000 bales of U.S. cotton worth about 110 million U.S. dollars [1].' | The end user | **It is what gets scored** |

### Why the key facts stay in telex style

Key facts are copied verbatim from the article, so `evals.checks` can verify automatically that every expected article contains every key fact. Rewriting them in normal English ('million') would break that check, because the article says 'mln'.

### How the checks bridge the two styles

| Check | Key fact '39.3 mln N.Z. Dlr', answer '39.3 million New Zealand dollars' |
|---|---|
| Verbatim (string match) | ✗: 'mln' is not 'million' |
| Judge (Claude, per fact) | ✓: reads the fact for its meaning, and quotes 'cash bid of 39.3 million New Zealand dollars' as evidence |
| Normalized (abbreviations written one way) | ✓: both sides become '39.3 mln new zealand dlr' before matching. Matched 53 of the 73 facts the verbatim check missed, so fewer verdicts rest on the judge alone ([findings](evaluation.md#2026-10-20-normalising-abbreviations-closes-most-of-the-verbatim-gap)) |

The key facts say *what* a correct answer must contain; the checks decide whether an answer contains it, in whatever words. Nothing about the user's answer has to change for the evaluation, and the evaluation never changes what the user sees: it measures answers offline, after they are given.

## 2. The evaluation as a CI gate on Databricks

The golden dataset is a regression test: after every change to prompts, models or retrieval settings, the same questions show whether the app still behaves correctly. This is a typical setup with Databricks' usual building blocks; the details depend on the team.

```
 Developer changes the prompt, model, top_k, ...
        │
        ▼
 ┌─────────────────────── Pull request (GitHub) ───────────────────────┐
 │                                                                     │
 │  (1) Fast checks (GitHub Actions, every push, free, ~1 min)         │
 │     ruff · evals.checks (dataset rules) · unit tests                │
 │                                                                     │
 │  (2) Evaluation job (Databricks Job, started from GitHub Actions)   │
 │     │                                                               │
 │     │  golden dataset              (repo file or Unity Catalog      │
 │     │  questions + key facts        table)                          │
 │     ▼                                                               │
 │   For each question, the RAG pipeline of THIS branch:               │
 │     filter extraction → Vector Search index → LLM answer            │
 │                         (exists already, built once by an           │
 │                          ingest job)                                │
 │     Each question becomes an MLflow trace.                          │
 │     │                                                               │
 │     ▼                                                               │
 │   Retrieval metrics: hit@1, hit@k, MRR    (evals.evaluate)          │
 │   mlflow.genai.evaluate with the scorers  (evals.score_answers):    │
 │     · key_facts_verbatim   code check, free, deterministic          │
 │     · key_facts_normalized the same, abbreviations written one way  │
 │     · fact_coverage        LLM judge (Claude), one verdict per      │
 │                            key fact, with a quote as evidence       │
 │     · groundedness         MLflow's built-in LLM judge (Claude):    │
 │                            every claim supported by the passages?   │
 │     │                                                               │
 │     ├──► MLflow experiment: run, metrics, every trace + verdicts    │
 │     ▼                                                               │
 │   Gate: compare with thresholds, or with main's last runs           │
 │     e.g. hit@4 ≥ 90%, all facts (judge) ≥ 90%, grounded ≥ 95%       │
 │     │                                                               │
 │     ├── below ──► ✗ PR check fails: read the traces in MLflow,      │
 │     │             fix, push again (back to (1))                     │
 │     ▼                                                               │
 │   ✓ PR check passes                                                 │
 └─────┼───────────────────────────────────────────────────────────────┘
       ▼
 Merge to main ──► deploy (Asset Bundle) ──► Model Serving endpoint
                                                 │
                                                 ▼
                                     Real users ask questions
                                                 │
 ┌─────────────── Production monitoring ─────────┼─────────────────────┐
 │   Every request is an MLflow trace            ▼                     │
 │   The same judges (groundedness, ...) score a sample of traffic.    │
 │   Thumbs-down and failed questions ──► reviewed ──► added to the    │
 │                                                    golden dataset ──┼──► next PR
 └─────────────────────────────────────────────────────────────────────┘
```

### The steps

**① Fast checks, on every push.** Free, deterministic, no API key and no index needed. They catch broken code and broken cases: `ruff`, `evals.checks`, unit tests. This repo runs the first two in GitHub Actions ([.github/workflows/ci.yml](../.github/workflows/ci.yml)); `evals.checks` uses only the standard library, so the job installs nothing but ruff and caches the Reuters archive between runs.

**② The evaluation job.** The same two scripts as now, `evals.evaluate` and `evals.score_answers`. This repo runs them as a Databricks job with a gate on thresholds, started by every pull request that changes code; how it is set up is in [databricks-setup.md](databricks-setup.md#12-evaluation-job). On Databricks, three things change:

| | Now (local) | On Databricks |
|---|---|---|
| Where results go | `sqlite:///mlflow.db` | The workspace's MLflow: only the tracking URI changes |
| Vector index | Chroma in `data/`, git-ignored, built locally | Databricks Vector Search, which already exists: no rebuilding per run |
| How it starts | By hand | A Databricks Job, started from the pull request |

Every run costs API calls: the questions are fixed, but the answers are generated fresh, because the answers of the changed app are what is being tested. That is cents for the answers and tens of cents for the judges per run of 50 questions. This is why the evaluation runs per pull request (or nightly), not per push.

**The gate.** The one new part. Today a person reads the numbers; in CI a script compares them with thresholds and fails the check below them. LLM steps vary by a few points between identical runs (the judge's 'all facts' ranged 92–96% over three runs), so a threshold sits somewhat below the baseline, or the gate compares with the average of several runs on main. A failed gate does not say what broke; the traces in MLflow do, with every retrieval, answer and verdict.

**Production monitoring.** After deployment, real requests are traced too, and the same scorers judge a sample of them. This answers a different question from the golden dataset:

| | Golden dataset in CI | Production monitoring |
|---|---|---|
| Question | Did my change break something? | Does the live system still work? |
| Data | Fixed, reviewed questions with known answers | Real questions, no known answers |
| Scorers | All: needs key facts for the fact checks | Only those without expectations, e.g. groundedness |

**The loop back.** Real questions are harder than generated ones ([evaluation.md](evaluation.md#7-pitfalls)). Questions that fail in production, or get a thumbs-down, are reviewed like the first 50 cases (question, expected articles, key facts) and added to the golden dataset. Each becomes a test for the next change, so the dataset grows where the system is weak.
