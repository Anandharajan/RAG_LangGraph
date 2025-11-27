from langchain_core.tools import tool

def get_retriever_tool(vectorstore):
    """
    Creates a LangChain tool from the vector store retriever.
    """
    retriever = vectorstore.as_retriever()
    
    @tool
    def retrieve_rag_docs(query: str) -> str:
        """Search and retrieve information about the RAG Chatbot and LangGraph Agent project from the knowledge base."""
        # Use invoke if available, else get_relevant_documents
        if hasattr(retriever, "invoke"):
            docs = retriever.invoke(query)
        else:
            docs = retriever.get_relevant_documents(query)
            
        return "\n\n".join([d.page_content for d in docs])
    
    return retrieve_rag_docs
