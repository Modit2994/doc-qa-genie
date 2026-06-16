"""
ingest.py — Document parsing, chunking, embedding, and ChromaDB storage.

Supported formats : PDF (.pdf), Word (.docx)
Max file size     : 2 MB (enforced by caller in app.py)
"""

from __future__ import annotations

import io
import os
from pathlib import Path

# On corporate/restricted networks the HF hub causes proxy errors.
# Only force offline mode when NOT running on Streamlit Cloud.
_on_streamlit_cloud = os.environ.get("HOME", "").startswith("/home/adminuser")
if not _on_streamlit_cloud:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from typing import Optional

import chromadb
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer

from chunker import Chunk, chunk_document

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_DIR = Path(__file__).parent / ".chroma"
COLLECTION_NAME = "doc_qa"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MAX_FILE_BYTES = 2 * 1024 * 1024  # 2 MB

# ---------------------------------------------------------------------------
# Singleton helpers — model and chroma client are loaded once per session
# ---------------------------------------------------------------------------

_embedder: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        try:
            # Use locally cached model first — avoids proxy/firewall issues
            _embedder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        except Exception:
            # Fallback: allow network if cache is missing (first-time setup)
            _embedder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=False)
    return _embedder


def get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        CHROMA_DIR.mkdir(exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client


def get_collection(reset: bool = False) -> chromadb.Collection:
    client = get_chroma_client()
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_pdf(file_bytes: bytes) -> list[tuple[int, str]]:
    """Return list of (page_number, page_text) from a PDF byte stream."""
    pages: list[tuple[int, str]] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append((page_num, text))
    return pages


def _parse_docx(file_bytes: bytes) -> list[tuple[int, str]]:
    """
    Return list of (virtual_page, text) from a .docx byte stream.
    Word documents don't have hard page numbers, so we assign a
    virtual page every ~50 paragraphs.
    """
    doc = DocxDocument(io.BytesIO(file_bytes))
    pages: list[tuple[int, str]] = []
    buffer: list[str] = []
    virtual_page = 1

    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            buffer.append(text)
        if len(buffer) >= 50:
            pages.append((virtual_page, "\n".join(buffer)))
            buffer = []
            virtual_page += 1

    if buffer:
        pages.append((virtual_page, "\n".join(buffer)))

    return pages


# ---------------------------------------------------------------------------
# Main ingest function
# ---------------------------------------------------------------------------

def ingest_document(
    file_bytes: bytes,
    filename: str,
    reset: bool = True,
) -> dict:
    """
    Parse → chunk → embed → store in ChromaDB.

    Returns a summary dict with chunk_count, strategy used, etc.
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        pages = _parse_pdf(file_bytes)
    elif ext == ".docx":
        pages = _parse_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Only .pdf and .docx are accepted.")

    if not pages:
        raise ValueError("No text could be extracted from the document.")

    # Embed (needed both for semantic chunking and for vector storage)
    embedder = get_embedder()

    # Chunk — passes embedder so semantic strategy can be attempted
    chunks, strategy = chunk_document(pages, source_file=filename, embedder=embedder)

    if not chunks:
        raise ValueError("Document produced no chunks after processing.")

    texts = [c.text for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    # Reset collection on the first file; append for subsequent files
    collection = get_collection(reset=reset)

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": c.source_file,
            "page": c.page,
            "chunk_index": c.chunk_index,
            "section": c.section or "",
        }
        for c in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    return {
        "filename": filename,
        "total_pages": len(pages),
        "chunk_count": len(chunks),
        "strategy": strategy,
    }
