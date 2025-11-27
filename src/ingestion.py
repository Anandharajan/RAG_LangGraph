from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import CHUNK_SIZE, CHUNK_OVERLAP

def load_pdf(file_path):
    """
    Loads a PDF file and returns a list of documents.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def chunk_documents(documents):
    """
    Splits documents into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def ingest_file(file_path):
    """
    Orchestrates loading and chunking.
    """
    docs = load_pdf(file_path)
    chunks = chunk_documents(docs)
    print(f"Loaded {len(docs)} pages and created {len(chunks)} chunks.")
    return chunks
