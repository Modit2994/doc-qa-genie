"""
chunker.py — Three-tier chunking strategy:

  1. Heading-based (smart)   — when 3+ section headings are detected.
  2. Semantic (embedding)    — groups sentences by meaning similarity;
                               used when no clear heading structure exists.
  3. Fixed-size (fallback)   — 1000-word windows with 15% overlap;
                               used when semantic chunking yields too few
                               or too small chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Words-per-chunk target and overlap ratio for the fixed fallback
FIXED_CHUNK_WORDS = 1000
OVERLAP_RATIO = 0.15

# Minimum average words per chunk for semantic output to be accepted
_SEMANTIC_MIN_AVG_WORDS = 40
# Minimum number of chunks for semantic output to be accepted
_SEMANTIC_MIN_CHUNKS = 2


@dataclass
class Chunk:
    text: str
    page: int
    chunk_index: int
    section: Optional[str] = None
    source_file: str = ""
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Heading detection helpers
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(
    r"^(?:"
    r"#{1,6}\s+"                        # Markdown-style ## Heading
    r"|(?:\d+\.)+\s+\w"                 # Numbered  1.2 Section
    r"|Chapter\s+\d+"                   # Chapter N
    r"|PART\s+[IVXLC\d]+"              # PART IV
    r"|[A-Z][A-Z\s]{4,50}$"            # ALL-CAPS short line
    r")",
    re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    return bool(_HEADING_RE.match(stripped))


# ---------------------------------------------------------------------------
# Shared sub-split helper (prevents oversized chunks in any strategy)
# ---------------------------------------------------------------------------

def _subsplit_large_chunks(chunks: list[Chunk], source_file: str) -> list[Chunk]:
    """Further split any chunk that exceeds 1.5× FIXED_CHUNK_WORDS."""
    refined: list[Chunk] = []
    overlap = int(FIXED_CHUNK_WORDS * OVERLAP_RATIO)
    step = FIXED_CHUNK_WORDS - overlap

    for chunk in chunks:
        words = chunk.text.split()
        if len(words) > FIXED_CHUNK_WORDS * 1.5:
            sub_idx = 0
            for s in range(0, len(words), step):
                sub_text = " ".join(words[s: s + FIXED_CHUNK_WORDS])
                if sub_text.strip():
                    refined.append(
                        Chunk(
                            text=sub_text,
                            page=chunk.page,
                            chunk_index=chunk.chunk_index * 100 + sub_idx,
                            section=chunk.section,
                            source_file=source_file,
                        )
                    )
                sub_idx += 1
        else:
            refined.append(chunk)
    return refined


# ---------------------------------------------------------------------------
# Strategy 1 — Heading-based (smart) chunking
# ---------------------------------------------------------------------------

def _smart_chunk(pages: list[tuple[int, str]], source_file: str) -> list[Chunk]:
    """Split text on detected headings; each section becomes one or more chunks."""
    chunks: list[Chunk] = []
    current_heading: Optional[str] = None
    current_lines: list[str] = []
    current_page: int = 1

    def _flush(heading, lines, page, idx):
        text = "\n".join(lines).strip()
        if text:
            chunks.append(
                Chunk(
                    text=text,
                    page=page,
                    chunk_index=idx,
                    section=heading,
                    source_file=source_file,
                )
            )

    idx = 0
    for page_num, page_text in pages:
        for line in page_text.splitlines():
            if _looks_like_heading(line):
                _flush(current_heading, current_lines, current_page, idx)
                if current_lines:
                    idx += 1
                current_heading = line.strip()
                current_lines = []
                current_page = page_num
            else:
                current_lines.append(line)

    _flush(current_heading, current_lines, current_page, idx)
    return chunks


# ---------------------------------------------------------------------------
# Strategy 2 — Semantic chunking (embedding-similarity breakpoints)
# ---------------------------------------------------------------------------

def _semantic_chunk(
    pages: list[tuple[int, str]],
    source_file: str,
    embedder: "SentenceTransformer",
) -> list[Chunk]:
    """
    Embed each sentence, find similarity drops between consecutive sentences,
    and split at those breakpoints (adaptive threshold: mean − 1 std-dev).
    Returns an empty list if the result is not meaningful (triggers fallback).
    """
    import numpy as np

    # Collect sentences with their originating page number
    sentences: list[tuple[str, int]] = []
    for page_num, page_text in pages:
        for sent in _SENTENCE_SPLIT_RE.split(page_text.strip()):
            sent = sent.strip()
            # Skip very short fragments (noise)
            if len(sent.split()) >= 6:
                sentences.append((sent, page_num))

    if len(sentences) < _SEMANTIC_MIN_CHUNKS + 1:
        return []

    texts = [s[0] for s in sentences]

    # normalize_embeddings=True → dot product == cosine similarity (faster)
    embeddings = embedder.encode(
        texts, show_progress_bar=False, normalize_embeddings=True, batch_size=64
    )

    # Cosine similarities between consecutive sentence pairs
    similarities = [
        float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]

    # Adaptive breakpoint threshold
    mean_sim = float(np.mean(similarities))
    std_sim = float(np.std(similarities))
    threshold = mean_sim - std_sim

    # Group sentences into chunks at breakpoints
    chunks: list[Chunk] = []
    current_sents: list[tuple[str, int]] = [sentences[0]]
    idx = 0

    for i, sim in enumerate(similarities):
        next_sent = sentences[i + 1]
        if sim < threshold:
            text = " ".join(s[0] for s in current_sents)
            page = current_sents[0][1]
            chunks.append(
                Chunk(text=text, page=page, chunk_index=idx, source_file=source_file)
            )
            idx += 1
            current_sents = [next_sent]
        else:
            current_sents.append(next_sent)

    if current_sents:
        text = " ".join(s[0] for s in current_sents)
        page = current_sents[0][1]
        chunks.append(
            Chunk(text=text, page=page, chunk_index=idx, source_file=source_file)
        )

    # Validate quality — reject if chunks are too few or too small
    if len(chunks) < _SEMANTIC_MIN_CHUNKS:
        return []
    avg_words = sum(len(c.text.split()) for c in chunks) / len(chunks)
    if avg_words < _SEMANTIC_MIN_AVG_WORDS:
        return []

    return chunks


# ---------------------------------------------------------------------------
# Strategy 3 — Fixed-size fallback (word-based with overlap)
# ---------------------------------------------------------------------------

def _fixed_chunk(pages: list[tuple[int, str]], source_file: str) -> list[Chunk]:
    """Word-count-based chunking with overlap across pages."""
    word_list: list[tuple[str, int]] = []
    for page_num, page_text in pages:
        for word in page_text.split():
            word_list.append((word, page_num))

    overlap = int(FIXED_CHUNK_WORDS * OVERLAP_RATIO)
    step = FIXED_CHUNK_WORDS - overlap
    chunks: list[Chunk] = []

    start = 0
    idx = 0
    while start < len(word_list):
        end = min(start + FIXED_CHUNK_WORDS, len(word_list))
        segment = word_list[start:end]
        text = " ".join(w for w, _ in segment)
        page = segment[0][1] if segment else 1
        chunks.append(
            Chunk(text=text, page=page, chunk_index=idx, source_file=source_file)
        )
        idx += 1
        start += step

    return chunks


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def chunk_document(
    pages: list[tuple[int, str]],
    source_file: str = "",
    embedder: "Optional[SentenceTransformer]" = None,
) -> tuple[list[Chunk], str]:
    """
    Three-tier chunking. Returns (chunks, strategy_label).

    Priority:
      1. Heading-based  — when 3+ headings detected in the document.
      2. Semantic        — embedding-similarity breakpoints; requires embedder.
      3. Fixed fallback  — 1000-word windows with 15% overlap.
    """
    all_text = "\n".join(t for _, t in pages)
    lines = all_text.splitlines()
    heading_count = sum(1 for line in lines if _looks_like_heading(line))

    # ── Strategy 1: heading-based ──────────────────────────────────────────
    if heading_count >= 3:
        chunks = _smart_chunk(pages, source_file)
        chunks = _subsplit_large_chunks(chunks, source_file)
        return chunks, "heading-based (smart)"

    # ── Strategy 2: semantic (embedding) ──────────────────────────────────
    if embedder is not None:
        try:
            sem_chunks = _semantic_chunk(pages, source_file, embedder)
            if sem_chunks:
                sem_chunks = _subsplit_large_chunks(sem_chunks, source_file)
                return sem_chunks, "semantic (embedding-based)"
        except Exception:
            pass  # silently fall through to fixed

    # ── Strategy 3: fixed fallback ─────────────────────────────────────────
    pct = int(OVERLAP_RATIO * 100)
    return _fixed_chunk(pages, source_file), f"fixed (1000 words, {pct}% overlap)"
