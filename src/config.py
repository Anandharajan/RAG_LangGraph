import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"

# Data Paths
PDF_PATH = DATA_DIR / "source.pdf"  # We will rename the input PDF to this
VECTORSTORE_PATH = DATA_DIR / "faiss_index"

# RAG Parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM Parameters (Hugging Face free Inference API)
# Default router model should exist on the router. Override via HF_MODEL_ID env var or UI input.
# Meta Llama 3 8B Instruct is widely available on the HF router as of Nov 2024.
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")  # Optional for many free endpoints
LOCAL_MODEL_ID = os.getenv("LOCAL_MODEL_ID", "distilgpt2")
TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.3"))
