import gradio as gr
import os
import shutil
from src.config import PDF_PATH, HF_API_TOKEN, HF_MODEL_ID, DATA_DIR
from src.ingestion import ingest_file
from src.vectorstore import create_vectorstore, load_vectorstore
from src.agent import build_langgraph_agent
from langchain_core.messages import HumanMessage

# Global variables to store state
vectorstore = None
agent_executor = None
current_hf_token = None
current_hf_model = None

# Ensure data directory exists for uploads and FAISS index (important for HF Spaces).
os.makedirs(DATA_DIR, exist_ok=True)


def _get_uploaded_path(uploaded_file):
    """
    Normalize Gradio's uploaded file into a filesystem path.
    Handles filepath strings, temporary file objects, and dict payloads.
    """
    if uploaded_file is None:
        return None

    if isinstance(uploaded_file, (str, os.PathLike)):
        return str(uploaded_file)

    if isinstance(uploaded_file, dict):
        return uploaded_file.get("name") or uploaded_file.get("path")

    if hasattr(uploaded_file, "name"):
        return uploaded_file.name

    return None


def initialize_system(hf_token, hf_model, uploaded_file):
    """
    Initializes the RAG pipeline and Agent.
    """
    global vectorstore, agent_executor, current_hf_token, current_hf_model
    
    hf_token = (hf_token or HF_API_TOKEN or "").strip()
    hf_model = (hf_model or HF_MODEL_ID).strip()
    uploaded_path = _get_uploaded_path(uploaded_file)
    
    if uploaded_file is not None and uploaded_path is None:
        return "Could not read the uploaded file. Please try uploading again."

    if uploaded_path is None and not os.path.exists(PDF_PATH):
        return "Please upload a PDF file."

    try:
        # 0. Handle File Upload
        if uploaded_path is not None:
            # Gradio passes a temporary file path or a file object depending on version/config.
            # Usually it's a named temp file path in recent versions.
            # We copy it to our data directory.
            if not os.path.exists(os.path.dirname(PDF_PATH)):
                os.makedirs(os.path.dirname(PDF_PATH))
                
            # uploaded_file is a file path in recent Gradio versions
            shutil.copy(uploaded_path, PDF_PATH)
            print(f"File saved to {PDF_PATH}")
            
            # Force re-ingestion since we have a new file
            print("Ingesting PDF...")
            chunks = ingest_file(str(PDF_PATH))
            vectorstore = create_vectorstore(chunks)
        
        # 1. Load or Create Vector Store (if not already created above)
        if vectorstore is None:
            vectorstore = load_vectorstore()
            if vectorstore is None:
                # This case should be covered by the upload logic, but just in case
                if os.path.exists(PDF_PATH):
                    print("Ingesting PDF...")
                    chunks = ingest_file(str(PDF_PATH))
                    vectorstore = create_vectorstore(chunks)
                else:
                    return "Source PDF not found. Please upload a file."
        
        # 2. Create Agent (LangGraph)
        agent_executor = build_langgraph_agent(vectorstore, hf_api_token=hf_token, hf_model_id=hf_model)
        current_hf_token = hf_token
        current_hf_model = hf_model
        mode = "Hugging Face router" if hf_token else "local transformers (no HF token provided)"
        
        return f"System Initialized Successfully using {mode}. You can now start chatting."
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Initialization Failed: {str(e)}"

def chat(message, history, hf_token, hf_model, uploaded_file):
    """
    Chat function for Gradio.
    """
    global agent_executor, current_hf_token, current_hf_model
    
    # Gradio can pass None for history on the first turn.
    history = history or []
    if not message:
        return "Please enter a message to start chatting."
        
    hf_token = (hf_token or HF_API_TOKEN or "").strip()
    hf_model = (hf_model or HF_MODEL_ID).strip()
    
    # Check if API key has changed or agent is not initialized
    if agent_executor is None or hf_token != current_hf_token or hf_model != current_hf_model:
        init_msg = initialize_system(hf_token, hf_model, uploaded_file)
        if "Failed" in init_msg or "Please" in init_msg:
            return init_msg
    
    # Run the agent
    try:
        # Convert history to LangChain format if needed, but LangGraph handles state.
        # We pass the full history + new message to the agent if we were managing state manually,
        # but here we'll just pass the new message and let the graph handle it if we were persistent.
        # For a simple chat interface without persistence, we pass the conversation history.
        
        messages = []
        for h in history:
            messages.append(HumanMessage(content=h[0]))
            # We would need AI message here too, but Gradio history is [user, bot].
            # For simplicity in this demo, we'll just send the current message or a limited context.
            # Let's send the current message. To support history, we'd need to map Gradio history to LangChain messages.
        
        # Better approach for this demo: Just send the current message. 
        # The agent is stateless between calls in this simple implementation unless we use checkpointers.
        
        response = agent_executor.invoke({"messages": [HumanMessage(content=message)]})
        return response["messages"][-1].content
    except Exception as e:
        import traceback
        traceback.print_exc()
        hint = (
            " If you used the Hugging Face router, verify the token/model. "
            "Otherwise, try re-initializing to refresh the vector store."
        )
        return f"Error while generating a reply: {str(e)}{hint}"

# Gradio UI
with gr.Blocks(title="RAG Chatbot (LangGraph + HF)") as demo:
    gr.Markdown("# RAG-Based Chatbot (LangGraph + Hugging Face)")
    gr.Markdown(
        "Upload a PDF, build a vector store, retrieve context, and answer with either the Hugging Face router "
        "(when a token + router model is provided) or a local fallback model."
    )
    
    with gr.Row():
        api_key_input = gr.Textbox(
            label="Hugging Face API Token (optional)", 
            type="password", 
            placeholder="hf_...",
            value=os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
        )
        model_input = gr.Textbox(
            label="Model ID",
            placeholder="e.g. meta-llama/Meta-Llama-3-8B-Instruct",
            value=os.getenv("HF_MODEL_ID", HF_MODEL_ID),
        )
        file_input = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
        init_btn = gr.Button("Initialize System")
    
    status_output = gr.Textbox(label="Status", interactive=False)
    
    chatbot = gr.ChatInterface(
        fn=chat, 
        additional_inputs=[api_key_input, model_input, file_input]
    )

    init_btn.click(initialize_system, inputs=[api_key_input, model_input, file_input], outputs=[status_output])

if __name__ == "__main__":
    # Use local launch by default; share links can fail without network access.
    demo.queue().launch(
        share=False,
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
