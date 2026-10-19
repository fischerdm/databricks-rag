import logging
import uuid
from pathlib import Path

from chainlit.utils import mount_chainlit
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import RedirectResponse, StreamingResponse

from app.config import get_settings
from app.schemas import DocumentInput, IndexResponse, QueryInput, SearchFilters
from app.services import llm, retrieval, vector

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Backend API",
    description="FastAPI backend for document indexing and a RAG chatbot",
    version="1.0.0",
)


def index_document_task(document_id: str, content: str, metadata: dict) -> None:
    try:
        n_chunks = vector.add_document(document_id, content, metadata)
        logger.info("Indexed document %s (%d chunks)", document_id, n_chunks)
    except Exception:
        logger.exception("Failed to index document %s", document_id)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse("/ui")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/index", status_code=202, response_model=IndexResponse)
async def index_document(doc: DocumentInput, background_tasks: BackgroundTasks) -> IndexResponse:
    """
    Accept text and index it into the vector database in the background,
    so the client does not have to wait for processing.
    """
    document_id = str(uuid.uuid4())
    background_tasks.add_task(index_document_task, document_id, doc.content, doc.metadata)
    return IndexResponse(
        status="processing",
        document_id=document_id,
        message="Document has been queued for indexing.",
    )


@app.post("/query")
async def query_rag(payload: QueryInput) -> StreamingResponse:
    """
    Retrieve relevant context and stream the LLM's answer back token by token (SSE).

    Optional `date_from`, `date_to` and `topic` restrict the search before ranking by similarity.
    With `auto_filters` (default), missing filters are extracted from the question by an LLM.
    """
    top_k = payload.top_k or get_settings().top_k
    explicit = SearchFilters(**payload.model_dump(include=set(SearchFilters.model_fields)))
    applied, chunks = await retrieval.retrieve(payload.question, top_k, explicit, payload.auto_filters)

    return StreamingResponse(
        llm.stream_answer(payload.question, chunks, applied),
        media_type="text/event-stream",
    )


# Chainlit chat UI, served from the same process; it calls /query like any other client.
mount_chainlit(app=app, target=str(Path(__file__).with_name("ui.py")), path="/ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
