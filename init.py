"""
init.py — One-time initialization script.

Run this ONCE after installing requirements to:
  1. Download and cache the embedding model (BAAI/bge-small-en-v1.5, ~130 MB)
  2. Verify Ollama is reachable and list available models
  3. Create the local ChromaDB storage directory

Usage:
    python init.py
"""

import sys


def check_python_version():
    if sys.version_info < (3, 10):
        print(f"❌ Python 3.10+ required. You are running {sys.version}")
        print("   Please use the conda environment: conda activate docqa")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]}")


def download_embedding_model():
    print("\n📦 Downloading embedding model (BAAI/bge-small-en-v1.5) ...")
    print("   This is ~130 MB and only happens once. Subsequent runs use the cache.")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        # Quick smoke-test
        _ = model.encode(["hello world"])
        print("✅ Embedding model ready and cached.")
    except Exception as e:
        print(f"❌ Failed to download embedding model: {e}")
        sys.exit(1)


def check_ollama():
    print("\n🦙 Checking Ollama ...")
    try:
        import ollama
        models = ollama.list()
        model_names = [m.model for m in models.models]
        if model_names:
            print(f"✅ Ollama is running. Available models: {', '.join(model_names)}")
        else:
            print(
                "⚠️  Ollama is running but no models are pulled yet.\n"
                "    Pull a model before using the app:\n"
            "      ollama pull phi3:mini      (default, 3.8B, low RAM)\n"
            "      ollama pull llama3.1      (8B, best quality, needs 8 GB RAM)\n"
            "      ollama pull mistral        (7B, fast)\n"
            )
    except Exception:
        print(
            "⚠️  Could not reach Ollama. Make sure it is installed and running:\n"
            "      https://ollama.com\n"
            "    Start it with:  ollama serve\n"
            "    Then pull a model: ollama pull llama3.1"
        )


def init_chroma():
    print("\n🗄️  Initialising ChromaDB storage ...")
    try:
        from pathlib import Path
        import chromadb
        db_path = Path(__file__).parent / ".chroma"
        db_path.mkdir(exist_ok=True)
        client = chromadb.PersistentClient(path=str(db_path))
        client.get_or_create_collection("doc_qa")
        print(f"✅ ChromaDB ready at {db_path}")
    except Exception as e:
        print(f"❌ ChromaDB initialisation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 50)
    print("  Doc Q&A — Initialisation")
    print("=" * 50)

    check_python_version()
    download_embedding_model()
    check_ollama()
    init_chroma()

    print("\n" + "=" * 50)
    print("✅  All set! Run the app with:")
    print("     streamlit run app.py")
    print("=" * 50)
