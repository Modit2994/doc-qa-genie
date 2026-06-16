# Doc Q&A — Ask Questions from Any Document

Upload a PDF or Word file and ask questions about it in plain English.
Get instant, accurate answers with page-level source citations.

> **No cloud storage. Your documents stay on your machine.**

---

## What Does This App Do?

- Upload **up to 3** PDF or `.docx` files at once (max 2 MB each) for deep cross-document Q&A
- Type a question like *"What are the payment terms?"* or *"Summarise section 3"*
- Get a streamed answer with the exact page and section it came from, citing which document
- Ask follow-up questions — the app remembers the conversation

---

## Two Ways to Use It — Pick What Works for You

| | Option A: Groq API ☁️ | Option B: Local (Ollama) 🔒 |
|---|---|---|
| **Setup time** | ~2 minutes | ~15 minutes + model download |
| **Internet needed** | Yes (only for the answer, not your file) | No — fully offline |
| **Answer quality** | Higher (LLaMA 3.1 8B on Groq cloud) | Good (phi3:mini on your laptop) |
| **Your document sent outside?** | No — only small matched text snippets | No — nothing leaves your machine |
| **Best for** | Quick setup, better accuracy | Sensitive documents, offline use |

### Recommended starting point → **Option A: Groq API**

Groq is a free cloud service that runs open-source AI models at high speed.
You get a free API key in under 2 minutes — no credit card required.

---

## Quick Start — Groq API (Recommended)

### Step 1 — Get a Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com) and sign up (Google or email)
2. Click **API Keys** → **Create API Key**
3. Name it anything (e.g. `doc-qa`) and copy the key — it starts with `gsk_`

> The key is free. Groq does not charge for typical document Q&A usage.

### Step 2 — Install Python Dependencies

You need Python 3.11+. If you don't have it, install
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) first (see Appendix).

```bash
cd doc-qa
pip install -r requirements.txt
python init.py        # one-time setup (~2 min, downloads embedding model)
```

### Step 3 — Run the App

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

### Step 4 — Use the App

1. In the sidebar → **LLM Mode** → select **API (Groq)**
2. Paste your `gsk_...` key into the **Groq API Key** field
3. Click **Test API Key** to confirm it works
4. Upload your PDF or Word file
5. Ask your question in the chat box

> **Save your key permanently** so you don't paste it every session:
> ```bash
> cp .env.example .env
> # Open .env and replace gsk_your_key_here with your actual key
> ```

---

## Alternative — Fully Offline (Local Ollama)

Use this if you need **complete privacy** (nothing leaves the machine at all)
or if you're working without internet.

### Step 1 — Install Ollama

Download from [ollama.com](https://ollama.com/download) and install.

### Step 2 — Pull a Model

Open Terminal and run one of these (choose based on your laptop's RAM):

| Your RAM | Command | Notes |
|---|---|---|
| 16 GB+ | `ollama pull llama3.1` | Best answer quality |
| 8 GB | `ollama pull mistral` | Fast and accurate |
| 4–8 GB | `ollama pull phi3:mini` | Already done if you followed this guide |
| < 4 GB | `ollama pull gemma2:2b` | Minimum viable |

### Step 3 — Start Ollama

```bash
ollama serve
```

Keep this terminal open while using the app.

### Step 4 — Run the App

```bash
streamlit run app.py
```

In the sidebar, keep **LLM Mode** set to **Local (Ollama)**.
Your pulled model appears automatically in the dropdown.

---

## One-Time Setup Detail (`python init.py`)

Run this once after `pip install`. It:
- Downloads and caches the text embedding model (~130 MB, free, open-source)
- Checks if Ollama is running (skippable if using Groq)
- Creates the local database folder

You will see something like:
```
✅ Python 3.11.15
✅ Embedding model ready and cached.
⚠️  Ollama not running — that's fine if you're using Groq API mode.
✅ ChromaDB ready
✅  All set! Run the app with: streamlit run app.py
```

---

## Complete Setup Checklist

**Groq API path (recommended):**
```
[ ] pip install -r requirements.txt
[ ] python init.py
[ ] Get free key at console.groq.com
[ ] streamlit run app.py  →  paste key in sidebar
```

**Local Ollama path:**
```
[ ] pip install -r requirements.txt
[ ] python init.py
[ ] Install Ollama + pull a model
[ ] ollama serve  (keep running in separate terminal)
[ ] streamlit run app.py
```

---

## Project Files

```
doc-qa/
├── app.py              # Main UI — upload (up to 3 files), chat, streaming responses
├── ingest.py           # Reads documents → chunks → embeds → stores locally
├── retriever.py        # Finds the most relevant chunks for your question
├── llm.py              # Sends question to Ollama or Groq and streams the answer
├── chunker.py          # Three-tier chunking: heading-based / semantic / fixed
├── init.py             # One-time setup script
├── requirements.txt    # Python dependencies
├── .env.example        # Template for saving your Groq API key
└── README.md
```

---

## How Chunking Works (Plain English)

When you upload a document, the app automatically picks the best splitting strategy:

| Priority | Strategy | Used when |
|---|---|---|
| **1 — Heading-based** | Splits by detected section headings | Document has 3+ clear headings |
| **2 — Semantic** | Groups sentences by meaning using AI embeddings; cuts where topic changes | No clear headings but text has paragraph structure |
| **3 — Fixed fallback** | 1000-word windows with 15% overlap | Unstructured / scan-like documents |

The sidebar shows which strategy was picked for your file. Oversized chunks are always further split automatically.

This means the app searches *relevant sections* rather than the whole document — making answers faster and more precise.

---

## Privacy Summary

| What happens to your document | Local mode | Groq API mode |
|---|---|---|
| Full document stored anywhere online | Never | Never |
| Full document sent to an AI service | Never | Never |
| Small matching text snippets sent out | Never | Yes, to Groq only |
| Document stored on your machine | Yes (in `.chroma/` folder) | Yes (in `.chroma/` folder) |

---

## Limitations

- Maximum **3 files** per session (uploading a new set replaces the previous one)
- Maximum **2 MB per file**
- Text only — images and tables within documents are not extracted
- `.pdf` and `.docx` formats only

---

## Appendix — Installing Python 3.11 via Miniconda

macOS includes Python 3.9 by default which is too old. Miniconda installs a
newer version in your home folder without needing admin access.

**Apple Silicon (M1/M2/M3) Mac:**
```bash
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o miniconda.sh
bash miniconda.sh -b -p ~/miniconda3 && rm miniconda.sh
~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
mkdir -p ~/miniconda3/envs
~/miniconda3/bin/conda create -p ~/miniconda3/envs/docqa python=3.11 -y
~/miniconda3/bin/conda init zsh   # or bash
# Restart terminal, then:
conda activate ~/miniconda3/envs/docqa
```

**Intel Mac:**
```bash
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p ~/miniconda3 && rm miniconda.sh
# same steps as above from conda tos accept onwards
```

---

## License

Open source — for personal and non-commercial use.
