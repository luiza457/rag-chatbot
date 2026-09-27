from __future__ import annotations

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.document_loader import SUPPORTED_EXTENSIONS
from app.embeddings import build_embedder
from app.generation import AnswerGenerator
from app.service import RAGService
from app.vector_store import FaissVectorStore

logger = logging.getLogger(__name__)
service: RAGService | None = None


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=10)


def get_service() -> RAGService:
    if service is None:
        raise HTTPException(status_code=503, detail="RAG service is not ready")
    return service


@asynccontextmanager
async def lifespan(_: FastAPI):
    global service
    embedder = build_embedder(settings.embedding_backend, settings.embedding_model)
    store = FaissVectorStore(embedder.dimension, settings.data_dir)
    loaded = store.load()
    generator = AnswerGenerator(settings.openai_api_key, settings.openai_model)
    service = RAGService(settings, embedder, store, generator)
    if not loaded and settings.auto_ingest_sample and settings.documents_dir.exists():
        count = service.ingest_directory(settings.documents_dir)
        logger.info("Automatically indexed %s sample chunks", count)
    yield
    service = None


app = FastAPI(
    title="Technical Documentation RAG",
    version="1.0.0",
    description="Grounded question answering over private technical documents.",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health() -> dict:
    rag = get_service()
    return {
        "status": "ok",
        "indexed_chunks": rag.store.size,
        "embedding_backend": settings.embedding_backend,
        "generation_mode": rag.generator.mode,
    }


@app.post("/ask")
def ask(request: QuestionRequest) -> dict:
    return get_service().ask(request.question, request.top_k)


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Supported file types: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    with NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
        shutil.copyfileobj(file.file, temporary)
        path = Path(temporary.name)
    try:
        count = get_service().ingest_file(path)
        # The temporary filename would be unhelpful in citations, so update source labels.
        store = get_service().store
        for index in range(len(store.chunks) - count, len(store.chunks)):
            chunk = store.chunks[index]
            store.chunks[index] = type(chunk)(
                id=chunk.id,
                text=chunk.text,
                source=file.filename or "uploaded_document",
                page=chunk.page,
                chunk_index=chunk.chunk_index,
            )
        if hasattr(store, "save"):
            store.save()
        return {"filename": file.filename, "chunks_added": count}
    finally:
        path.unlink(missing_ok=True)

