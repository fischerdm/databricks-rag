# fastapi-rag-boilerplate

A minimal RAG backend: index text over an API, then ask questions and get a streamed answer grounded in the indexed documents.

- **API:** FastAPI
- **Embeddings / LLM:** OpenAI (`text-embedding-3-small`, `gpt-4o-mini` by default)
- **Vector store:** Chroma (persisted locally in `data/chroma`)

How it works, component by component: [docs/architecture.md](docs/architecture.md).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then set OPENAI_API_KEY
.venv/bin/uvicorn app.main:app --reload
```

Interactive docs: http://127.0.0.1:8000/docs

## Corpus: Reuters-21578

[Reuters-21578](https://archive.ics.uci.edu/dataset/137/reuters+21578+text+categorization+collection) is a classic collection of Reuters business and financial newswire articles from Feb–Oct 1987. The ingest script downloads it into `data/reuters21578/` and indexes the ~19k articles that have a body:

```bash
.venv/bin/python -m scripts.ingest_reuters --limit 500   # quick sample
.venv/bin/python -m scripts.ingest_reuters               # full corpus
```

Each chunk carries `title`, `date`, `date_int` (e.g. `19870315`, for range filters), `topics` and `places` (lists of Reuters codes) as metadata. Re-running the script updates existing entries instead of duplicating them.

To refresh metadata on an existing index without re-embedding (e.g. after a metadata format change), run `python -m scripts.ingest_reuters --metadata-only`.

## Endpoints

| Method | Path      | Description                                                        |
|--------|-----------|--------------------------------------------------------------------|
| GET    | `/health` | Liveness check                                                     |
| POST   | `/index`  | Queue a document for chunking + embedding; returns `202` and an id |
| POST   | `/query`  | Retrieve the top-k chunks and stream the answer as SSE             |

```bash
curl -X POST localhost:8000/index -H 'Content-Type: application/json' \
  -d '{"content": "ECB holds rates steady...", "metadata": {"source": "example"}}'

curl -N -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question": "What did the ECB decide?"}'
```

`/query` accepts optional filters that restrict the search before ranking by similarity:

```bash
curl -N -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question": "What happened to oil prices?", "date_from": "1987-10-19", "date_to": "1987-10-20", "topic": "crude"}'
```

`topic` is a Reuters topic code such as `crude`, `acq` (acquisitions), `earn` (earnings) or `money-fx`; the codes that occur in the corpus are listed in [app/topics.py](app/topics.py).

### Automatic filters

By default (`"auto_filters": true`) an LLM (`FILTER_MODEL`, default `gpt-5.4-nano`) reads the question first and fills in missing filters, using OpenAI structured outputs with a Pydantic model as the response format:

| Question | Extracted filters |
|---|---|
| What happened to oil prices on October 19, 1987? | 1987-10-19 – 1987-10-20, `crude` |
| Which takeovers were announced in March? | 1987-03-01 – 1987-03-31, `acq` |
| How did the dollar trade against the yen in early April 1987? | 1987-04-01 – 1987-04-10, `money-fx` |

- Explicit filters in the request always win; dates are taken as a pair.
- A single extracted day is widened by one day, since news about an event often runs the day after.
- If the extraction fails, or its filters match nothing, the search falls back to the explicit filters only.

`/query` emits four event types: `filters` (the filters actually applied, once), `sources` (the retrieved chunks, once), `token` (JSON-encoded text pieces), and `done`.
