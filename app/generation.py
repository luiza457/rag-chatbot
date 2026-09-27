from __future__ import annotations

from app.models import SearchResult


GROUNDING_INSTRUCTIONS = """You answer questions about private technical documents.
Use only the supplied context. Do not use outside knowledge or invent details.
Every factual statement must be supported by the context and cited with [1], [2], etc.
If the context is insufficient, say: "I could not find that in the indexed documents."
Keep the answer concise and practical."""


class AnswerGenerator:
    def __init__(self, api_key: str | None, model: str) -> None:
        self.api_key = api_key
        self.model = model

    @property
    def mode(self) -> str:
        return "openai" if self.api_key else "retrieval_only"

    def generate(self, question: str, results: list[SearchResult]) -> str:
        if not results:
            return "I could not find that in the indexed documents."

        context = "\n\n".join(
            f"[{number}] SOURCE: {result.chunk.source}"
            + (f", page {result.chunk.page}" if result.chunk.page else "")
            + f"\n{result.chunk.text}"
            for number, result in enumerate(results, start=1)
        )

        if not self.api_key:
            sources = ", ".join(f"[{i}]" for i in range(1, len(results) + 1))
            return (
                "Retrieval-only mode: the most relevant passages are returned below "
                f"as {sources}. Add OPENAI_API_KEY to generate a synthesised answer."
            )

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.responses.create(
            model=self.model,
            instructions=GROUNDING_INSTRUCTIONS,
            input=f"QUESTION:\n{question}\n\nCONTEXT:\n{context}",
        )
        return response.output_text.strip()

