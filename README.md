# RAG — Retrieval-Augmented Generation Project

project demonstrating document ingestion, embedding, and a FAISS-backed vector store
for retrieval-augmented generation (RAG).

## Overview

This repository loads documents from the `data/` folder, chunks and embeds them using
`sentence-transformers`, indexes the embeddings with FAISS, and exposes a simple search
and summarization flow using a Groq LLM (configured in `src/search.py`).

## Requirements

- Python 3.10+ (recommended)
- Install dependencies from `requirements.txt`:

```bash
uv add -r requirements.txt
```

## Files of interest

- `src/data_loader.py` — loads PDF/TXT/CSV/Word/JSON files from `data/`.
- `src/embedding.py` — chunks documents and builds embeddings with `sentence-transformers`.
- `src/vectorstore.py` — FAISS-backed `FaissVectorStore` implementation (build/load/query).
- `src/search.py` — RAG search wrapper that uses the vector store and an LLM to summarize.
- `app.py` — example runner that builds or loads the FAISS store and runs a sample query.

## Usage

1. Ensure your project dependencies are installed (see Requirements).
2. Place documents under the `data/` folder (the repository includes `data/text_files/` and `data/pdfs/`).
3. Run the example application from the project root:

```bash
python app.py
```

What the script does:
- Loads documents from `data/`.
- If a FAISS index isn't present, it will build the index from the documents and persist it.
- Loads the FAISS store and performs a sample search+summarization via the configured LLM.

## Persisted FAISS store

After running `app.py` once, a directory named `faiss_store` will be created in the project root.
It will contain at least the following files:

- `faiss.index` — the FAISS binary index file containing the vectors.
- `metadata.pkl` — a pickled Python list mapping each indexed vector to its document/chunk metadata.

These files allow the vector store to be reloaded without re-embedding all documents.

## Troubleshooting

- If `faiss` fails to import on Windows, install the CPU wheel for your Python version or use WSL.
- If embeddings are missing or empty, check that `data/` contains supported files and that
  `src/data_loader.py` finds them (it prints debug info when run).

## Next steps

- Configure the LLM API keys and model in `src/search.py`.
- Add more document types or adjust chunk sizes in `src/embedding.py`.

---