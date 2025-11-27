import os
from langchain_community.vectorstores import FAISS
try:
    # Preferred newer package
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback to older location if extra package is missing
    from langchain_community.embeddings import HuggingFaceEmbeddings
from .config import EMBEDDING_MODEL_NAME, VECTORSTORE_PATH

def get_embeddings():
    """
    Initializes the embedding model.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def create_vectorstore(chunks):
    """
    Creates a FAISS vector store from chunks and saves it locally.
    """
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(VECTORSTORE_PATH))
    return vectorstore

def load_vectorstore():
    """
    Loads the FAISS vector store from disk.
    """
    embeddings = get_embeddings()
    if os.path.exists(VECTORSTORE_PATH):
        return FAISS.load_local(str(VECTORSTORE_PATH), embeddings, allow_dangerous_deserialization=True)
    return None
