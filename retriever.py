"""
retriever.py — Similarity search against ChromaDB and context assembly
for the LLM prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ingest import get_collection, get_embedder

# Number of chunks to retrieve per query
TOP_K = 5


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    section: Optional[str]
    score: float  # cosine distance (lower = more similar)


def retrieve(query: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
    """
    Embed the query and return the top-K most similar chunks from ChromaDB.
    """
    embedder = get_embedder()
    query_embedding = embedder.encode([query], show_progress_bar=False).tolist()[0]

    collection = get_collection(reset=False)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[RetrievedChunk] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append(
            RetrievedChunk(
                text=doc,
                source=meta.get("source", ""),
                page=int(meta.get("page", 0)),
                section=meta.get("section") or None,
                score=round(dist, 4),
            )
        )

    return chunks


def build_context(chunks: list[RetrievedChunk]) -> str:
    """
    Assemble the retrieved chunks into a single context block
    for inclusion in the LLM prompt.
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        location = f"Page {chunk.page}"
        if chunk.section:
            location += f" | Section: {chunk.section}"
        parts.append(f"[Source {i} — {location}]\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def build_prompt(query: str, context: str) -> str:
    """
    Construct the final prompt sent to the LLM.
    Instructs the model to answer only from the provided context,
    avoid low-confidence guesses, and always cite sources.
    """
    return f"""Check the document context below and provide the most relevant answer to the user's question.

Rules you MUST follow:
1. Answer ONLY using the document context provided — do NOT use outside knowledge.
2. If you are not confident or the answer is not clearly present, respond with exactly: "I was unable to find a clear answer for this in the document." — do NOT give a low-confidence or speculative answer.
3. ALWAYS cite the source at the end of your answer in the format: [Page X] or [Page X — Section: Y].
4. Be concise and direct. Do not pad the answer.

--- DOCUMENT CONTEXT ---
{context}
--- END OF CONTEXT ---

User Question: {query}

Answer:"""
