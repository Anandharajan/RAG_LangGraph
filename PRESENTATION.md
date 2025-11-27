# RAG LangGraph Chatbot – Research Briefing

This document summarizes the implemented project, highlights differences versus the reference guide (“RAG-Based Chatbot with LangGraph Agent Integration.pdf”), and provides detailed notes for a technical presentation to a research audience.

## 1. Concept & Scope
- Goal: A Retrieval-Augmented Generation (RAG) chatbot that ingests a PDF, builds a vector store, retrieves context, and answers via a LangGraph-orchestrated flow exposed through a Gradio UI.
- Contrast with the PDF guide: The guide is CLI-first and modular (per-stage scripts); this project consolidates the flow into a single Gradio app while still following best practices (chunking, embeddings, FAISS, LangGraph orchestration).
- Design choice: Favor a minimal, reproducible web UI for demos and HF Spaces deployment rather than a set of separate CLI tools.

## 2. Key Differences vs. the Reference PDF
- **Interface**: PDF describes CLI pipelines; this repo uses a Gradio UI with one-click ingest + chat.
- **Orchestration**: Minimal LangGraph (retrieve → generate) instead of a richer tool/agent graph.
- **Models**: Default router model `meta-llama/Meta-Llama-3-8B-Instruct` (HF router) with local fallback `distilgpt2`; the PDF mentions OpenAI/local LLM options (Ollama, vLLM) not included here.
- **Vector store**: FAISS only (auto-create/load on upload); PDF discusses FAISS/Chroma and manual CLI scripts.
- **Ingestion**: PyPDFLoader + RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200) baked into the app.
- **Deployment**: Ready for Hugging Face Spaces (gradio sdk, space.yaml, runtime.txt); PDF assumes local CLI execution.
- **Validation**: Simple test in `tests/test_pipeline.py`; PDF suggests broader per-stage validation scripts.

## 3. Code Structure (files and roles)
- `app.py`: Gradio Blocks UI; handles upload, ingestion, vector store creation/loading, LangGraph agent invocation, HF token/model inputs.
- `src/config.py`: Paths, chunking params, embedding model, default HF router/local model IDs, temperature.
- `src/ingestion.py`: PDF loading (PyPDFLoader) and chunking (RecursiveCharacterTextSplitter).
- `src/vectorstore.py`: Embeddings via sentence-transformers/all-MiniLM-L6-v2; FAISS create/load.
- `src/agent.py`: LangGraph graph (retrieve node → generate node); HF router call with local fallback; prompt construction; context truncation for small local models.
- `src/rag_tool.py`: Legacy retriever tool helper (not wired into the app).
- `tests/test_pipeline.py`: Basic ingestion + FAISS save/load sanity check.
- Deployment metadata: `.gitignore`, `requirements.txt`, `runtime.txt`, `space.yaml`, README, PRESENTATION.md.

## 4. Libraries Used (requirements.txt, definitions & rationale)
- `langchain`, `langchain-community`, `langchain-text-splitters`, `langchain-huggingface`: Retrieval, prompt building, loaders, embeddings helpers.
- `langgraph`: Graph-based orchestration (stateful retrieve → generate flow).
- `gradio`: Web UI for chat and upload.
- `python-dotenv`: Load env vars for tokens/model IDs.
- `sentence-transformers`: Embedding model (all-MiniLM-L6-v2).
- `faiss-cpu`: Local vector index (fast similarity search).
- `pypdf`: PDF extraction for ingestion.
- `pydantic`: Settings/data validation utilities (dependency of LC ecosystem).
- `huggingface-hub`: Model hub interactions; router requests.
- `transformers`: HF router/local generation fallback pipeline.

## 5. End-to-End Flow (App)
1) Upload PDF (or reuse existing `data/source.pdf`).
2) Ingest → chunk (1000/200) → embed (all-MiniLM-L6-v2) → FAISS save (`data/faiss_index`).
3) Build LangGraph agent (retriever node, generator node).
4) Chat: user message → retrieve top docs → prompt with context → generate via HF router (token+model) else local fallback → return answer.

## 6. CLI Notes (for parity with PDF guide)
While the app is UI-first, equivalent stages can be executed manually:
```bash
# (Optional) venv, install
python -m venv .venv && .venv/Scripts/activate  # Windows; use bin/activate on *nix
pip install -r requirements.txt

# Ingest manually (scripted example)
python - <<'PY'
from src.ingestion import ingest_file
from src.vectorstore import create_vectorstore
from src.config import PDF_PATH
chunks = ingest_file(str(PDF_PATH))
_ = create_vectorstore(chunks)
print("Chunks:", len(chunks))
PY

# Run app
python app.py
```

## 7. Feasibility & Trade-offs
- **Router-first**: Best quality requires HF token + router model; local fallback is lightweight and will be weaker—set `LOCAL_MODEL_ID` to a stronger local model if needed (ensure it fits resources).
- **Minimal LangGraph**: Only two nodes (retrieve, generate). Extensible to add tools/checkpoints but kept lean for deployment stability.
- **Resource footprint**: Embedding model is small; FAISS is local and fast; defaults avoid GPU needs for embeddings, but generation quality depends on chosen LLM.
- **HF Spaces**: Configured with `space.yaml`/`runtime.txt` (Python 3.10) and Gradio sdk 4.44.1; tested build success.

## 8. Best Practices Applied
- Clear defaults with env override (`HF_MODEL_ID`, `LOCAL_MODEL_ID`, `HUGGINGFACEHUB_API_TOKEN`).
- Safe fallback: router errors degrade to local model with a visible note.
- Prompt truncation for local models to avoid context overrun errors.
- Persisted vector store in `data/faiss_index`; `data/` git-ignored; creation of `DATA_DIR` on startup.
- Deployment metadata for HF Spaces; `.gitignore` to keep repo clean.
- Basic test coverage for ingestion/index creation.
- Minimal, readable LangGraph wiring for transparency.

## 9. Recommendations (Future Enhancements)
- Add richer LangGraph tools (citation, summarization, multi-hop retrieval).
- Plug in stronger local models (e.g., via Ollama/vLLM) with gated selection.
- Expose per-turn retrieval metadata in UI (sources/attribution).
- Add per-stage CLI scripts (extract/chunk/embed/query) to mirror the PDF guide.
- Add evaluation harness (retrieval precision, answer faithfulness).

## 10. Quick Presentation Outline
1) Motivation: RAG for grounded answers; LangGraph for explicit flow; Gradio for usability.
2) Architecture: Ingest → Embed → FAISS → LangGraph (retrieve → generate) → HF router/local.
3) Implementation highlights: `ingestion.py`, `vectorstore.py`, `agent.py`, `app.py`.
4) Deployment: HF Spaces config; env vars; defaults and fallbacks.
5) Gaps vs. guide: UI-first vs CLI-first; minimal graph; single vector store; limited local LLM support.
6) Next steps: stronger local models, richer tools, evaluation, attribution in UI.
