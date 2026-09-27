from __future__ import annotations

import hashlib
import re

from app.models import Chunk


def clean_text(text: str) -> str:
    """Normalise whitespace while retaining paragraph boundaries."""
    text = text.replace("\x00", " ").replace("\r\n", "\n")
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in text.split("\n\n")]
    return "\n\n".join(p for p in paragraphs if p)


def chunk_text(
    text: str,
    source: str,
    page: int | None = None,
    max_words: int = 180,
    overlap_words: int = 30,
) -> list[Chunk]:
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must be between 0 and max_words - 1")

    words = clean_text(text).split()
    if not words:
        return []

    chunks: list[Chunk] = []
    step = max_words - overlap_words
    for chunk_index, start in enumerate(range(0, len(words), step)):
        piece = words[start : start + max_words]
        if not piece:
            break
        content = " ".join(piece)
        raw_id = f"{source}:{page}:{chunk_index}:{content}".encode("utf-8")
        chunks.append(
            Chunk(
                id=hashlib.sha256(raw_id).hexdigest()[:16],
                text=content,
                source=source,
                page=page,
                chunk_index=chunk_index,
            )
        )
        if start + max_words >= len(words):
            break
    return chunks

