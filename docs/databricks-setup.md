# Setting up the Databricks workspace

How the Databricks workspace for this project was set up: the account, the CLI, the network check, the raw data and the OpenAI key. Later steps (chunk table, Vector Search, the app) build on what is created here.

**Contents**

1. [Account](#1-account)
2. [CLI and login](#2-cli-and-login)
3. [What the workspace offers](#3-what-the-workspace-offers)
4. [Outbound network access](#4-outbound-network-access)
5. [Raw data in Unity Catalog](#5-raw-data-in-unity-catalog)
6. [OpenAI key as a secret](#6-openai-key-as-a-secret)
7. [Result](#7-result)

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

Next: an ingest job that parses and chunks the archive, embeds the chunks with OpenAI and writes them to a Delta table `workspace.rag.chunks`, deployed as a Databricks Asset Bundle.
