---
title: RAG LangGraph Chatbot
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
python_version: 3.10
---

---
title: RAG LangGraph Chatbot
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
python_version: "3.10"
---

# RAG-Based Chatbot (LangGraph + Hugging Face)

This project implements a RAG (Retrieval-Augmented Generation) chatbot that answers with either:
- **Hugging Face router** (when you provide an HF token and a router-available model; default `HF_MODEL_ID`: `meta-llama/Meta-Llama-3-8B-Instruct`), or
- **Local transformers generation** (no token; fallback `LOCAL_MODEL_ID`: `distilgpt2` by default — quality is limited; set a stronger local model if you need better offline answers).

## Features
- **RAG Pipeline**: Ingests, chunks, embeds, and indexes PDF documents for accurate retrieval.
- **Inference Flexibility**: Uses HF router when a token is provided; falls back to local transformers otherwise.
- **LangGraph Agent**: Retrieval + generation flow is orchestrated with LangGraph for clearer state handling.
- **Gradio Interface**: A user-friendly chat UI for interacting with the assistant.
- **Modular Design**: Clean separation of concerns (Ingestion, Vector Store, Agent, App).

## Project Structure
```
rag_agent_project/
├─ app.py              # Gradio application
├─ requirements.txt    # Dependencies
├─ data/               # Data storage (PDFs, Index)
├─ src/                # Source code
│  ├─ ingestion.py     # Data processing
│  ├─ vectorstore.py   # Embedding & Indexing
│  ├─ rag_tool.py      # (legacy) retriever tool helper
│  ├─ agent.py         # RAG + HF router/local agent
│  └─ config.py        # Configuration
└─ tests/              # Automated tests
```

## Setup & Usage

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure (optional)**:
    - Set `HUGGINGFACEHUB_API_TOKEN` for router inference.
    - Override `HF_MODEL_ID` for router (default: `meta-llama/Meta-Llama-3-8B-Instruct`).
    - Override `LOCAL_MODEL_ID` for local fallback (default: `distilgpt2`; use a stronger local model if you need better offline answers).

3.  **Run the Application**:
    ```bash
    python app.py
    ```

4.  **Interact**:
    - Open the provided local URL (usually `http://127.0.0.1:7860`).
    - (Optional) Provide a Hugging Face token and router-supported model ID for cloud inference (default: `meta-llama/Meta-Llama-3-8B-Instruct`).
    - Without a token, the app uses a local fallback model (`LOCAL_MODEL_ID`, default: `distilgpt2`; quality is limited—use router + token for good answers or set a stronger local model).
    - Upload a PDF and click "Initialize System".
    - Start chatting!

## Deployment (Hugging Face Spaces)
1.  Create a new Space on Hugging Face (SDK: Gradio).
2.  Upload the contents of `rag_agent_project` to the Space.
3.  Ensure `requirements.txt` is present.
4.  The app will build and launch automatically.

## Technical Details
- **LLM**: HF router (with token, default `meta-llama/Meta-Llama-3-8B-Instruct`) or local transformers fallback (`LOCAL_MODEL_ID`, default `distilgpt2`; change to a stronger model if running locally).
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS
- **Orchestration**: LangGraph (retrieve → generate) RAG prompt with retrieval context

## Notes for Hugging Face Spaces
- Add your `HUGGINGFACEHUB_API_TOKEN` as a secret for router usage.
- If you want to pin a different router model, set `HF_MODEL_ID` in the Space variables. Override `LOCAL_MODEL_ID` if you want a specific offline fallback.
- The `data/` folder is persisted for uploads and FAISS index; it is git-ignored here but created at runtime.
- Entry point is `app.py`; `demo.queue().launch()` is enabled for Spaces concurrency.
- Current status: build verified on HF Space `Anandharajan/RAG_LangGraph` with Python 3.10 (via `space.yaml`/`runtime.txt`).
