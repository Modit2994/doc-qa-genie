"""
llm.py — LLM client supporting two modes:

  LOCAL  : Ollama (fully offline, no API key needed)
  API    : Groq   (free tier, fast, open-source models on cloud)

Privacy note: In API mode only the retrieved document chunks
(~4-6 text snippets) are sent to Groq — the full document stays local.
"""

from __future__ import annotations

import os
import warnings
from typing import Generator

# Suppress SSL verification warnings produced by verify=False on corporate networks
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# ---------------------------------------------------------------------------
# Mode constants
# ---------------------------------------------------------------------------

MODE_LOCAL = "Local (Ollama)"
MODE_API   = "API (Groq)"

# ---------------------------------------------------------------------------
# Local — Ollama defaults
# ---------------------------------------------------------------------------

OLLAMA_DEFAULT_MODEL = "phi3:mini"

OLLAMA_SUGGESTED_MODELS = [
    "phi3:mini",    # 3.8B — default, low RAM
    "llama3.1",     # 8B  — best quality
    "mistral",      # 7B  — very fast
    "gemma2:2b",    # 2B  — ultra-light
]

# ---------------------------------------------------------------------------
# API — Groq defaults
# ---------------------------------------------------------------------------

GROQ_DEFAULT_MODEL = "llama-3.1-8b-instant"

GROQ_MODELS = [
    "llama-3.1-8b-instant",     # fast, free, great for Q&A
    "llama-3.3-70b-versatile",  # higher quality, still free tier
    "mixtral-8x7b-32768",       # large context window
    "gemma2-9b-it",             # Google Gemma 2
]


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def list_ollama_models() -> list[str]:
    """Return locally pulled Ollama model names; falls back to suggestions."""
    try:
        import ollama
        response = ollama.list()
        names = [m.model for m in response.models]
        return names if names else OLLAMA_SUGGESTED_MODELS
    except Exception:
        return OLLAMA_SUGGESTED_MODELS


def stream_ollama(prompt: str, model: str = OLLAMA_DEFAULT_MODEL) -> Generator[str, None, None]:
    try:
        import ollama
        for chunk in ollama.generate(model=model, prompt=prompt, stream=True):
            token = chunk.get("response", "")
            if token:
                yield token
    except Exception as exc:
        import ollama as _ollama
        if isinstance(exc, _ollama.ResponseError):
            yield f"\n\n⚠️ Ollama error: {exc.error}"
        else:
            yield f"\n\n⚠️ Could not reach Ollama. Is it running? ({exc})"


# ---------------------------------------------------------------------------
# Groq helpers
# ---------------------------------------------------------------------------

def stream_groq(
    prompt: str,
    model: str = GROQ_DEFAULT_MODEL,
    api_key: str = "",
) -> Generator[str, None, None]:
    """
    Stream a response from Groq.
    Only the prompt (containing retrieved chunks, NOT the full document)
    is sent to Groq's servers.
    """
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        yield "⚠️ No Groq API key provided. Enter your key in the sidebar."
        return

    try:
        import httpx
        from groq import Groq

        # Two network issues common on corporate laptops:
        # 1. Local proxy (e.g. Cursor tunnel on 127.0.0.1) blocks API traffic → proxy=None
        # 2. Corporate SSL inspection uses a custom CA Python doesn't trust → verify=False
        # Both are acceptable for a local dev/personal tool.
        http_client = httpx.Client(proxy=None, verify=False)
        client = Groq(api_key=key, http_client=http_client)

        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as exc:
        yield f"\n\n⚠️ Groq error: {exc}"


# ---------------------------------------------------------------------------
# Unified public entry point
# ---------------------------------------------------------------------------

def stream_response(
    prompt: str,
    mode: str = MODE_LOCAL,
    model: str = OLLAMA_DEFAULT_MODEL,
    api_key: str = "",
) -> Generator[str, None, None]:
    """
    Single streaming entry point for both modes.
    Compatible with Streamlit's st.write_stream().
    """
    if mode == MODE_API:
        yield from stream_groq(prompt, model=model, api_key=api_key)
    else:
        yield from stream_ollama(prompt, model=model)
