"""
chunker.py — Smart chunking with TOC/heading detection,
falling back to fixed-size with 10% overlap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# Words-per-chunk target and overlap ratio for the fixed fallback
FIXED_CHUNK_WORDS = 1000
OVERLAP_RATIO = 0.10


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

# Matches lines that look like section headings (e.g. "1. Introduction",
# "Chapter 3 — Scope", "## Background", "EXECUTIVE SUMMARY")
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


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    return bool(_HEADING_RE.match(stripped))


# ---------------------------------------------------------------------------
# Smart (section-aware) chunking
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
# Fixed-size fallback chunking (word-based with overlap)
# ---------------------------------------------------------------------------

def _fixed_chunk(pages: list[tuple[int, str]], source_file: str) -> list[Chunk]:
    """Word-count-based chunking with overlap across pages."""
    # Flatten all text with page boundary markers
    word_list: list[tuple[str, int]] = []  # (word, page_number)
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
        # Use the page of the first word in the chunk
        page = segment[0][1] if segment else 1
        chunks.append(
            Chunk(
                text=text,
                page=page,
                chunk_index=idx,
                source_file=source_file,
            )
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
) -> list[Chunk]:
    """
    pages: list of (page_number, page_text) tuples.
    Returns a list of Chunk objects.

    Strategy:
      1. If headings are detected → smart/section-aware split.
      2. Otherwise → fixed 1000-word chunks with 10% overlap.
    """
    all_text = "\n".join(t for _, t in pages)
    lines = all_text.splitlines()
    heading_count = sum(1 for line in lines if _looks_like_heading(line))

    # Use smart chunking only when there are enough headings to be meaningful
    if heading_count >= 3:
        chunks = _smart_chunk(pages, source_file)
        # If smart chunking produced very large chunks, further split them
        refined: list[Chunk] = []
        for chunk in chunks:
            words = chunk.text.split()
            if len(words) > FIXED_CHUNK_WORDS * 1.5:
                # Sub-split the oversized section
                overlap = int(FIXED_CHUNK_WORDS * OVERLAP_RATIO)
                step = FIXED_CHUNK_WORDS - overlap
                sub_idx = 0
                for s in range(0, len(words), step):
                    sub_text = " ".join(words[s : s + FIXED_CHUNK_WORDS])
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

    return _fixed_chunk(pages, source_file)
