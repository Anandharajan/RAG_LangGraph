import os
import sys
from pathlib import Path

import pytest

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import PDF_PATH
from src.ingestion import chunk_documents, load_pdf
from src.vectorstore import create_vectorstore, load_vectorstore


@pytest.fixture(scope="module")
def chunks():
    """Load and chunk the PDF if it exists, otherwise skip the tests."""

    if not os.path.exists(PDF_PATH):
        pytest.skip(f"Skipping ingestion test: {PDF_PATH} not found.")

    docs = load_pdf(str(PDF_PATH))
    assert docs, "No documents loaded"

    chunks = chunk_documents(docs)
    assert chunks, "No chunks created"
    return chunks


def test_ingestion(chunks):
    assert len(chunks) > 0


def test_vectorstore(chunks):
    vs = create_vectorstore(chunks)
    assert vs is not None, "Vector store creation failed"

    loaded_vs = load_vectorstore()
    assert loaded_vs is not None, "Vector store loading failed"
