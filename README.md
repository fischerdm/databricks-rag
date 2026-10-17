# fastapi-rag-boilerplate

A minimal RAG backend: index text over an API, then ask questions and get a streamed answer grounded in the indexed documents.

- **API:** FastAPI
- **Embeddings / LLM:** OpenAI (`text-embedding-3-small`, `gpt-4o-mini` by default)
- **Vector store:** Chroma (persisted locally in `data/chroma`)

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

Each chunk carries `title`, `date`, `date_int` (e.g. `19870315`, for range filters), `topics` and `places` as metadata. Re-running the script updates existing entries instead of duplicating them.

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

`/query` emits three event types: `sources` (the retrieved chunks, once), `token` (JSON-encoded text pieces), and `done`.
