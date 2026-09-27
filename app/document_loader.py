from __future__ import annotations

from pathlib import Path

from app.chunking import chunk_text
from app.models import Chunk

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_document(
    path: Path, max_words: int = 180, overlap_words: int = 30
) -> list[Chunk]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    if suffix == ".pdf":
        from pypdf import PdfReader

        chunks: list[Chunk] = []
        reader = PdfReader(str(path))
        for page_number, page in enumerate(reader.pages, start=1):
            chunks.extend(
                chunk_text(
                    page.extract_text() or "",
                    source=path.name,
                    page=page_number,
                    max_words=max_words,
                    overlap_words=overlap_words,
                )
            )
        return chunks

    return chunk_text(
        path.read_text(encoding="utf-8"),
        source=path.name,
        max_words=max_words,
        overlap_words=overlap_words,
    )


def load_directory(
    directory: Path, max_words: int = 180, overlap_words: int = 30
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            chunks.extend(load_document(path, max_words, overlap_words))
    return chunks

