"""Dependency-light end-to-end check for the core RAG retrieval logic."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.document_loader import load_directory
from app.embeddings import HashingEmbedder
from app.generation import AnswerGenerator
from app.service import RAGService
from app.vector_store import NumpyVectorStore


def main() -> None:
    embedder = HashingEmbedder()
    store = NumpyVectorStore(embedder.dimension)
    settings = Settings(
        embedding_backend="hashing",
        documents_dir=ROOT / "docs",
        data_dir=ROOT / "data",
    )
    service = RAGService(
        settings,
        embedder,
        store,
        AnswerGenerator(api_key=None, model="unused"),
    )
    chunks = load_directory(settings.documents_dir, max_words=100, overlap_words=20)
    service.ingest_chunks(chunks)
    result = service.ask(
        "What checks should be made when bearing vibration increases?", top_k=2
    )
    print(result["answer"])
    for source in result["sources"]:
        print(f"- {source['source']} score={source['score']}: {source['text'][:140]}...")


if __name__ == "__main__":
    main()

