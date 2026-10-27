# Setting up the Databricks workspace

How the Databricks workspace for this project was set up: the account, the CLI, the network check, the raw data and the OpenAI key, then the chunk table, the Vector Search index and the switch that points the app at it.

**Contents**

1. [Account](#1-account)
2. [CLI and login](#2-cli-and-login)
3. [What the workspace offers](#3-what-the-workspace-offers)
4. [Outbound network access](#4-outbound-network-access)
5. [Raw data in Unity Catalog](#5-raw-data-in-unity-catalog)
6. [OpenAI key as a secret](#6-openai-key-as-a-secret)
7. [Result](#7-result)
8. [Chunk table: the ingest job](#8-chunk-table-the-ingest-job)
9. [Vector Search](#9-vector-search)
10. [Pointing the app at Vector Search](#10-pointing-the-app-at-vector-search)

---

## 1. Account

The project runs on **Databricks Free Edition**: free with no end date, no credit card or cloud account needed, serverless compute only, with quotas on Vector Search endpoints, apps and daily compute. The 14-day free trial has no feature limits but expires, so it only suits finishing everything in one go.

Sign up at databricks.com under 'Try Databricks' → 'Free Edition' (email, Google or Microsoft). The workspace URL has the form `https://dbc-<id>.cloud.databricks.com`.

## 2. CLI and login

Install the Databricks CLI with Homebrew. Newer Homebrew versions refuse formulas from third-party taps until the tap is trusted; `databricks/tap` is Databricks' official tap.

```bash
brew tap databricks/tap
brew trust databricks/tap
brew install databricks
databricks --version          # v1.22.1 at the time of writing
```

Log in with OAuth. This opens a browser window and saves a profile in `~/.databrickscfg`; no token ends up in the repo.

```bash
databricks auth login --host https://<workspace-url>
databricks current-user me    # prints your user as JSON
```

In the browser consent screen, 'All APIs' and 'Allow longer-running access' are fine for a personal workspace: the CLI needs broad access to create indexes, upload data and deploy apps, and longer-running access lets it refresh its token instead of asking for a login about every hour. The token cache lives in `~/.databricks/token-cache.json`.

## 3. What the workspace offers

Read-only commands that show what is available:

```bash
databricks catalogs list
databricks warehouses list
databricks vector-search-endpoints list-endpoints
databricks serving-endpoints list
databricks apps list
```

In the Free Edition workspace:

| Component | Available | Role in this project |
|---|---|---|
| Catalog | `workspace` | Holds the `rag` schema |
| SQL warehouse | 'Serverless Starter Warehouse' (2X-Small) | Queries on the Delta tables |
| Vector Search | No endpoints yet | Replaces Chroma |
| Apps | None yet | Hosts FastAPI and Chainlit |
| Embedding models | `databricks-gte-large-en`, `databricks-bge-large-en`, `databricks-qwen3-embedding-0-6b` | Possible replacement for `text-embedding-3-small` |
| Chat models | `databricks-meta-llama-3-3-70b-instruct`, `databricks-gpt-oss-120b`/`-20b`, `databricks-llama-4-maverick`, Qwen, Gemma, DeepSeek | Possible replacement for `gpt-4o-mini` |

The migration keeps OpenAI at first and switches to Databricks models later, as a separate step. Moving from Chroma to Vector Search can already shift retrieval; changing the models at the same time would make a drop in the [golden dataset](evaluation.md) scores impossible to attribute.

## 4. Outbound network access

Free Edition limits which hosts serverless compute can reach. A one-off serverless job checked the hosts this project needs, with plain HTTPS requests and no credentials (an HTTP error status still means the host answered):

```python
import json, urllib.error, urllib.request

targets = {
    "openai": "https://api.openai.com/v1/models",
    "uci_archive": "https://archive.ics.uci.edu/",
    "pypi": "https://pypi.org/simple/",
}
results = {}
for name, url in targets.items():
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            results[name] = f"reachable (HTTP {r.status})"
    except urllib.error.HTTPError as e:
        results[name] = f"reachable (HTTP {e.code})"
    except Exception as e:
        results[name] = f"BLOCKED: {type(e).__name__}: {e}"
dbutils.notebook.exit(json.dumps(results))
```

Run it by importing it as a notebook and submitting it without a cluster spec, which makes it serverless:

```bash
databricks workspace import /Users/<you>/scratch/connectivity_check \
  --file connectivity_check.py --language PYTHON --format SOURCE
databricks jobs submit --json '{"run_name": "connectivity-check",
  "tasks": [{"task_key": "check",
             "notebook_task": {"notebook_path": "/Users/<you>/scratch/connectivity_check"}}]}'
databricks jobs get-run-output <task-run-id>
```

| Host | Result | Consequence |
|---|---|---|
| `api.openai.com` | Reachable (HTTP 401 without a key) | OpenAI can stay for embeddings and answers |
| `pypi.org` | Reachable (HTTP 200) | Jobs and apps can install packages |
| `archive.ics.uci.edu` | Blocked (name does not resolve) | The Reuters archive is downloaded locally and uploaded (next section) |

## 5. Raw data in Unity Catalog

A schema for everything this project creates, and a managed volume for the raw archive:

```bash
databricks schemas create rag workspace \
  --comment "RAG chatbot: Reuters-21578 corpus, chunks and vector index"
databricks volumes create workspace rag raw MANAGED --comment "Raw Reuters-21578 archive"
```

Download [`reuters21578.tar.gz`](https://archive.ics.uci.edu/dataset/137/reuters+21578+text+categorization+collection) (7.8 MB) into `data/` on your machine (`data/*` is git-ignored) and upload it unchanged:

```bash
databricks fs cp data/reuters21578.tar.gz dbfs:/Volumes/workspace/rag/raw/reuters21578.tar.gz
databricks fs ls dbfs:/Volumes/workspace/rag/raw/ -l   # 8150596 bytes
```

The archive is unpacked and parsed on Databricks, by the ingest job, from `/Volumes/workspace/rag/raw/reuters21578.tar.gz`.

## 6. OpenAI key as a secret

The key lives in a Databricks secret scope, not in code or `.env`:

```bash
databricks secrets create-scope rag
databricks secrets put-secret rag openai_api_key   # prompts for the value
databricks secrets list-secrets rag
```

`list-secrets` shows only the key name and its last update time in milliseconds since 1970 (for example `1793088669058`), never the value. Code on Databricks reads it with `dbutils.secrets.get("rag", "openai_api_key")`, and notebook output shows it as `[REDACTED]`. A Databricks App gets it as an environment variable through a secret resource.

The key is not tested at this point; a wrong value shows up as an HTTP 401 on the first embedding call and is fixed by running `put-secret` again.

## 7. Result

| Object | Name |
|---|---|
| Schema | `workspace.rag` |
| Volume | `workspace.rag.raw` |
| Raw archive | `/Volumes/workspace/rag/raw/reuters21578.tar.gz` |
| Secret scope / key | `rag` / `openai_api_key` |

## 8. Chunk table: the ingest job

Everything from here on is defined in a Databricks Asset Bundle, [databricks.yml](../databricks.yml), and deployed with one command. The bundle sets no workspace host; it uses the CLI's default profile.

```bash
databricks bundle deploy
databricks bundle run ingest_chunks --params limit=200   # sample
databricks bundle run ingest_chunks                      # whole corpus
```

[jobs/ingest_chunks.py](../jobs/ingest_chunks.py) runs on serverless compute. It reads the archive from the volume, parses and chunks it with the same code as the local ingest (`scripts.ingest_reuters.iter_articles`, `app.services.chunking.chunk_text`), embeds each chunk with OpenAI (key from the `rag` scope) and writes one row per chunk to `workspace.rag.chunks`, with Change Data Feed enabled for Vector Search. Each run replaces the table's rows. The full corpus takes about 5 minutes: 19,043 articles, 28,261 chunks.

Two serverless details shaped the job: the script runs through `exec()` without `__file__`, so the bundle passes the repo root as `--repo-root ${workspace.file_path}`; and `--python-params` on `bundle run` would replace all task parameters, so the sample size is a job parameter (`limit`, `0` = everything).

The job appears under **Jobs & Pipelines** as '[dev <user>] rag-ingest-chunks' (development mode adds the prefix), the table under **Catalog** → `workspace` → `rag`.

## 9. Vector Search

The bundle also defines a standard endpoint `rag` and a Delta Sync index `workspace.rag.chunks_index` on the chunk table, using the job's embeddings (`embedding`, 1,536 dimensions) rather than computing its own. The endpoint takes several minutes to come up on the first deploy. The index syncs on request, so after an ingest run:

```bash
databricks vector-search-indexes sync-index workspace.rag.chunks_index
```

If a deploy is interrupted while it waits for the index, check `databricks bundle plan`: when it wants to create the index again, `databricks bundle deployment bind chunks_index workspace.rag.chunks_index` links the existing one.

## 10. Pointing the app at Vector Search

`VECTOR_BACKEND=databricks` (in `.env` or the environment) makes search and lookups go to the index instead of Chroma, through [app/services/vector_databricks.py](../app/services/vector_databricks.py). Locally, the Databricks SDK authenticates with the CLI profile; query embeddings still come from OpenAI. Indexing over `POST /index` is not available with this backend: the ingest job writes the data.

Vector Search ranks by L2 distance and reports `score = 1 / (1 + L2²)`. OpenAI embeddings have unit length, so `(1 / score − 1) / 2` is the cosine distance Chroma reports; distances, and with them `TOPIC_BOOST`, mean the same on both backends. Numbers come back as floats and are converted back to integers.

Retrieval on the [golden dataset](evaluation.md) (`python -m evals.evaluate --reviewed-only`, 50 cases, top 4):

| | Chroma | Vector Search |
|---|---|---|
| Without filter extraction: hit@1 / hit@4 / MRR | 88% / 96% / 0.92 | 88% / 96% / 0.92 |
| With filter extraction (`TOPIC_MODE=off`): hit@1 / hit@4 / MRR | 88% / 94% / 0.91 | 88% / 94% / 0.91 |

The six cases not ranked first are the same on both, with the same articles in the same order.
