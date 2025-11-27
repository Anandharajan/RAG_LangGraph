import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ingestion import load_pdf, chunk_documents
from src.vectorstore import create_vectorstore, load_vectorstore
from src.config import PDF_PATH

def test_ingestion():
    print("Testing Ingestion...")
    if not os.path.exists(PDF_PATH):
        print(f"Skipping ingestion test: {PDF_PATH} not found.")
        return
    
    docs = load_pdf(str(PDF_PATH))
    assert len(docs) > 0, "No documents loaded"
    print(f"Loaded {len(docs)} pages.")
    
    chunks = chunk_documents(docs)
    assert len(chunks) > 0, "No chunks created"
    print(f"Created {len(chunks)} chunks.")
    return chunks

def test_vectorstore(chunks):
    print("Testing Vector Store...")
    if not chunks:
        print("Skipping vector store test: No chunks.")
        return

    vs = create_vectorstore(chunks)
    assert vs is not None, "Vector store creation failed"
    print("Vector store created and saved.")
    
    loaded_vs = load_vectorstore()
    assert loaded_vs is not None, "Vector store loading failed"
    print("Vector store loaded successfully.")

if __name__ == "__main__":
    try:
        chunks = test_ingestion()
        test_vectorstore(chunks)
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
