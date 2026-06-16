"""
app.py — Streamlit UI for the Document Q&A app.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from ingest import MAX_FILE_BYTES, ingest_document
from llm import (
    GROQ_DEFAULT_MODEL,
    GROQ_MODELS,
    MODE_API,
    MODE_LOCAL,
    OLLAMA_DEFAULT_MODEL,
    list_ollama_models,
    stream_response,
)
from retriever import build_context, build_prompt, retrieve

# Load .env file if present (for GROQ_API_KEY)
load_dotenv()

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Doc Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

defaults = {
    "chat_history": [],
    "docs_meta": [],
    "llm_mode": MODE_LOCAL,
    "selected_model": OLLAMA_DEFAULT_MODEL,
    "groq_api_key": os.environ.get("GROQ_API_KEY", ""),
    "groq_model": GROQ_DEFAULT_MODEL,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📄 Doc Q&A")
    st.caption("Ask questions from any PDF or Word document.")

    st.divider()

    # ── 1. Upload ──────────────────────────────────────────────────────────
    st.subheader("1. Upload Documents")
    uploaded_files = st.file_uploader(
        label="PDF or Word files — up to 3 files, max 2 MB each",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Supported: .pdf, .docx  |  Max size: 2 MB each  |  Max 3 files",
    )

    if uploaded_files:
        if len(uploaded_files) > 3:
            st.error("Please upload a maximum of 3 files at a time.")
        else:
            # Only re-ingest when the set of filenames changes
            current_names = [f.name for f in uploaded_files]
            cached_names = [m["filename"] for m in st.session_state.docs_meta]
            if current_names != cached_names:
                st.session_state.docs_meta = []
                st.session_state.chat_history = []
                all_ok = True
                for i, uf in enumerate(uploaded_files):
                    file_bytes = uf.read()
                    if len(file_bytes) > MAX_FILE_BYTES:
                        st.error(
                            f"**{uf.name}** is too large "
                            f"({len(file_bytes) / 1024 / 1024:.1f} MB). Max 2 MB."
                        )
                        all_ok = False
                        break
                    with st.spinner(f"Processing {uf.name}… ({i + 1}/{len(uploaded_files)})"):
                        try:
                            meta = ingest_document(
                                file_bytes, uf.name, reset=(i == 0)
                            )
                            st.session_state.docs_meta.append(meta)
                        except ValueError as e:
                            st.error(f"**{uf.name}**: {e}")
                            all_ok = False
                            break

                if all_ok and st.session_state.docs_meta:
                    total_chunks = sum(m["chunk_count"] for m in st.session_state.docs_meta)
                    for meta in st.session_state.docs_meta:
                        st.success(
                            f"✅ **{meta['filename']}**\n\n"
                            f"- Pages: {meta['total_pages']}\n"
                            f"- Chunks: {meta['chunk_count']}\n"
                            f"- Strategy: {meta['strategy']}"
                        )
                    if len(st.session_state.docs_meta) > 1:
                        st.info(f"🔍 Deep check active — {total_chunks} total chunks across {len(st.session_state.docs_meta)} documents.")
            else:
                for meta in st.session_state.docs_meta:
                    st.success(
                        f"✅ **{meta['filename']}** — {meta['chunk_count']} chunks"
                    )

    st.divider()

    # ── 2. LLM Mode ────────────────────────────────────────────────────────
    st.subheader("2. LLM Mode")

    prev_mode = st.session_state.llm_mode
    st.session_state.llm_mode = st.radio(
        label="Choose LLM source",
        options=[MODE_LOCAL, MODE_API],
        index=0 if st.session_state.llm_mode == MODE_LOCAL else 1,
        help=(
            f"**{MODE_LOCAL}**: Fully offline via Ollama. Nothing leaves your machine.\n\n"
            f"**{MODE_API}**: Groq free-tier API. Only matched document chunks are sent — "
            "not the full document."
        ),
    )
    # Clear chat when switching modes to avoid stale error messages
    if st.session_state.llm_mode != prev_mode:
        st.session_state.chat_history = []

    if st.session_state.llm_mode == MODE_LOCAL:
        # Ollama model picker
        available = list_ollama_models()
        if st.session_state.selected_model not in available:
            available = [st.session_state.selected_model] + available
        st.session_state.selected_model = st.selectbox(
            label="Model (via Ollama)",
            options=available,
            index=available.index(st.session_state.selected_model),
            help=(
                "These are models already pulled on your machine via Ollama.\n\n"
                "To add more: `ollama pull llama3.1` or `ollama pull mistral`"
            ),
        )
        st.caption(
            f"🔒 Running **{st.session_state.selected_model}** locally via Ollama — "
            "no data leaves your machine."
        )

    else:
        # Groq API key + model picker
        api_key_input = st.text_input(
            label="Groq API Key",
            value=st.session_state.groq_api_key,
            type="password",
            placeholder="gsk_...",
            help="Get a free key at https://console.groq.com",
        )
        if api_key_input:
            # Strip invisible unicode / non-ASCII chars that browsers sometimes
            # inject when copying from a web page (causes ASCII codec errors)
            st.session_state.groq_api_key = (
                api_key_input.encode("ascii", errors="ignore").decode("ascii").strip()
            )

        # Show masked key info so user can verify it's correct
        if st.session_state.groq_api_key:
            k = st.session_state.groq_api_key
            st.caption(f"Key loaded: `{k[:8]}...{k[-4:]}` ({len(k)} chars)")

        # Test API Key button
        if st.button("🔍 Test API Key", use_container_width=True, disabled=not st.session_state.groq_api_key):
            with st.spinner("Testing key…"):
                try:
                    import httpx
                    from groq import Groq as _Groq
                    _client = _Groq(
                        api_key=st.session_state.groq_api_key,
                        http_client=httpx.Client(proxy=None, verify=False),
                    )
                    _models = _client.models.list()
                    st.success(f"✅ Key is valid! {len(_models.data)} models available.")
                except Exception as _e:
                    err = str(_e)
                    if "401" in err or "invalid_api_key" in err:
                        st.error("❌ Invalid API key. Please check you copied the full key from console.groq.com")
                    elif "Connection" in err:
                        st.error(f"❌ Connection failed: {err[:120]}")
                    else:
                        st.error(f"❌ {err[:150]}")

        st.session_state.groq_model = st.selectbox(
            label="Groq model",
            options=GROQ_MODELS,
            index=GROQ_MODELS.index(st.session_state.groq_model)
            if st.session_state.groq_model in GROQ_MODELS
            else 0,
        )

        st.caption(
            "⚠️ **Privacy:** Only the matched text chunks from your document "
            "are sent to Groq — not the full document.\n\n"
            "Get a free API key: [console.groq.com](https://console.groq.com)"
        )

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.title("📄 Document Q&A")

# ── Groq setup guide — shown only when API mode is active and key is missing ──
if st.session_state.llm_mode == MODE_API and not st.session_state.groq_api_key:
    st.warning("☁️ **Groq API mode is active but no API key is set.**")
    with st.expander("📋 How to get your free Groq API key — click to expand", expanded=True):
        st.markdown(
            """
**Groq is free** — no credit card needed. Takes about 60 seconds to set up.

**Step 1 — Create an account**
Go to [console.groq.com](https://console.groq.com) and sign up with Google or email.

**Step 2 — Generate an API key**
1. Click **API Keys** in the left sidebar
2. Click **Create API Key**
3. Give it a name (e.g. `doc-qa`) and copy the key — it starts with `gsk_`

**Step 3 — Add it to the app**
Paste the key into the **Groq API Key** field in the sidebar on the left.

---

> **Privacy reminder:** Only the small text chunks that match your question
> are sent to Groq — your full document always stays on your machine.

---

*Prefer fully offline?* Switch back to **Local (Ollama)** in the sidebar.
            """
        )

if not st.session_state.docs_meta:
    st.info("👈 Upload up to 3 PDF or Word documents from the sidebar to get started.")
    st.stop()

docs_meta = st.session_state.docs_meta
total_chunks = sum(m["chunk_count"] for m in docs_meta)

# Active docs + mode badge
if st.session_state.llm_mode == MODE_LOCAL:
    mode_label = f"🔒 Local · `{st.session_state.selected_model}`"
    privacy_note = ""
else:
    mode_label = f"☁️ Groq · `{st.session_state.groq_model}`"
    privacy_note = "  |  ⚠️ Matched chunks sent to Groq"

if len(docs_meta) == 1:
    doc_label = f"📁 **{docs_meta[0]['filename']}** — {total_chunks} chunks"
else:
    names = ", ".join(f"**{m['filename']}**" for m in docs_meta)
    doc_label = f"📂 {names} — {total_chunks} chunks across {len(docs_meta)} docs"

st.caption(f"{doc_label}  |  {mode_label}{privacy_note}")

st.divider()

# Render chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
query = st.chat_input("Ask a question about the document…")

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Searching document…"):
        chunks = retrieve(query)

    if not chunks:
        answer = "I could not find relevant information in the document for your question."
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
    else:
        context = build_context(chunks)
        prompt = build_prompt(query, context)

        # Determine active model label for the answer
        active_model = (
            st.session_state.selected_model
            if st.session_state.llm_mode == MODE_LOCAL
            else st.session_state.groq_model
        )

        with st.chat_message("assistant"):
            response_text = st.write_stream(
                stream_response(
                    prompt=prompt,
                    mode=st.session_state.llm_mode,
                    model=active_model,
                    api_key=st.session_state.groq_api_key,
                )
            )

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response_text}
        )

        # Source citations
        with st.expander("📎 Source chunks used", expanded=False):
            for i, chunk in enumerate(chunks, start=1):
                loc = f"Page {chunk.page}"
                if chunk.section:
                    loc += f" — {chunk.section}"
                st.markdown(f"**[{i}] {loc}** *(score: {chunk.score})*")
                st.text(chunk.text[:400] + ("…" if len(chunk.text) > 400 else ""))
                st.divider()
