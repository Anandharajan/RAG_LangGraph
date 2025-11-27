from typing import List, Optional, TypedDict
from types import SimpleNamespace
import requests
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from .config import HF_MODEL_ID, HF_API_TOKEN, LOCAL_MODEL_ID, TEMPERATURE

# Cache local model/pipeline to avoid repeated downloads.
_LOCAL_PIPELINE = None
_LOCAL_MODEL_ID = None


def _build_prompt(question: str, docs: List) -> str:
    """Create a concise prompt that uses retrieved context."""
    context = "\n\n".join(d.page_content for d in docs[:4])
    return (
        "You are a helpful assistant. Use the provided context to answer the question. "
        "If the context is insufficient, say you do not know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


class ChatState(TypedDict):
    messages: List[BaseMessage]
    context: str


def _hf_generate(prompt: str, model_id: str, token: Optional[str], temperature: float) -> str:
    """
    Minimal text generation call against the Hugging Face router API.
    """
    url = f"https://router.huggingface.co/models/{model_id}"
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": temperature,
            "return_full_text": False,
        },
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
    except requests.HTTPError as http_err:
        status = http_err.response.status_code if http_err.response is not None else None
        if status == 404:
            raise RuntimeError(
                f"Model '{model_id}' not found on Hugging Face router. "
                f"Set HF_MODEL_ID to a router-available text-generation model and retry."
            ) from http_err
        raise
    except requests.RequestException as req_err:
        # Network layer issues (timeouts, DNS, etc.) should surface cleanly so we can fall back.
        raise RuntimeError(f"Hugging Face router request failed: {req_err}") from req_err
    data = resp.json()
    # HF router can return list or dict; handle both
    if isinstance(data, list) and data and isinstance(data[0], dict):
        if "generated_text" in data[0]:
            return data[0]["generated_text"]
        if "error" in data[0]:
            raise RuntimeError(data[0]["error"])
    if isinstance(data, dict):
        if "generated_text" in data:
            return data["generated_text"]
        if "error" in data:
            raise RuntimeError(data["error"])
    return str(data)


def _local_generate(prompt: str, model_id: str, temperature: float) -> str:
    """
    Fallback local generation using transformers pipeline (no HF API token needed).
    Truncates the prompt to fit within the model's max position embeddings to avoid index errors.
    """
    global _LOCAL_PIPELINE, _LOCAL_MODEL_ID

    if _LOCAL_PIPELINE is None or _LOCAL_MODEL_ID != model_id:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        _LOCAL_PIPELINE = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="cpu",
        )
        _LOCAL_MODEL_ID = model_id

    tokenizer = _LOCAL_PIPELINE.tokenizer
    model = _LOCAL_PIPELINE.model
    max_new_tokens = 128

    # Determine max prompt length to prevent IndexError for small context windows (e.g., gpt2 = 1024).
    max_positions = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    pad_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    if max_positions and isinstance(max_positions, int):
        allowed = max_positions - max_new_tokens - 1
        if allowed > 0:
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(input_ids) > allowed:
                # Keep the tail of the prompt (most recent question + context)
                input_ids = input_ids[-allowed:]
                prompt = tokenizer.decode(input_ids, skip_special_tokens=True)

    outputs = _LOCAL_PIPELINE(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        pad_token_id=pad_token_id,
    )
    # transformers pipeline returns list of dicts
    if outputs and isinstance(outputs[0], dict) and "generated_text" in outputs[0]:
        return outputs[0]["generated_text"]
    return str(outputs)


def build_agent(
    vectorstore,
    hf_model_id: Optional[str] = None,
    hf_api_token: Optional[str] = None,
    temperature: Optional[float] = None,
):
    """
    Simple RAG agent using Hugging Face router inference (text_generation).
    """
    retriever = vectorstore.as_retriever()
    model_id = (hf_model_id or HF_MODEL_ID).strip()
    local_model_id = (LOCAL_MODEL_ID or model_id).strip()
    token = (hf_api_token or HF_API_TOKEN or "").strip() or None
    temp = TEMPERATURE if temperature is None else temperature

    def invoke(payload):
        messages = payload.get("messages", [])
        user_content = messages[-1].content if messages else ""

        # prefer invoke to avoid deprecation warnings
        if hasattr(retriever, "invoke"):
            docs = retriever.invoke(user_content)
        else:
            docs = retriever.get_relevant_documents(user_content)
        prompt = _build_prompt(user_content, docs)
        # Use router if a token is provided; otherwise fall back to local generation.
        try:
            if token:
                text = _hf_generate(prompt, model_id=model_id, token=token, temperature=temp)
            else:
                text = _local_generate(prompt, model_id=local_model_id, temperature=temp)
        except Exception as api_err:
            if token:
                # Degrade gracefully to local generation when router is flaky or the model is blocked.
                fallback_note = (
                    f"[Fallback to local model '{local_model_id}' because HF router failed: {api_err}]"
                )
                print(fallback_note)
                text = _local_generate(prompt, model_id=local_model_id, temperature=temp)
                text = f"{text}\n\n{fallback_note}"
            else:
                raise
        return {"messages": [AIMessage(content=text)]}

    # Return an object with an invoke method to mirror previous agent_executor shape
    return SimpleNamespace(invoke=invoke)


def build_langgraph_agent(
    vectorstore,
    hf_model_id: Optional[str] = None,
    hf_api_token: Optional[str] = None,
    temperature: Optional[float] = None,
):
    """
    LangGraph-based RAG agent with retrieval + generation nodes.
    """
    retriever = vectorstore.as_retriever()
    model_id = (hf_model_id or HF_MODEL_ID).strip()
    local_model_id = (LOCAL_MODEL_ID or model_id).strip()
    token = (hf_api_token or HF_API_TOKEN or "").strip() or None
    temp = TEMPERATURE if temperature is None else temperature

    def retrieve_node(state: ChatState):
        messages = state.get("messages", [])
        user_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        query = user_msg.content if user_msg else ""

        if hasattr(retriever, "invoke"):
            docs = retriever.invoke(query)
        else:
            docs = retriever.get_relevant_documents(query)
        context = "\n\n".join(d.page_content for d in docs[:4])
        return {"context": context}

    def generate_node(state: ChatState):
        messages = state.get("messages", [])
        context = state.get("context", "")
        user_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        question = user_msg.content if user_msg else ""

        prompt = (
            "You are a helpful assistant. Use the provided context to answer the question. "
            "If the context is insufficient, say you do not know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )

        try:
            if token:
                text = _hf_generate(prompt, model_id=model_id, token=token, temperature=temp)
            else:
                text = _local_generate(prompt, model_id=local_model_id, temperature=temp)
        except Exception as api_err:
            if token:
                fallback_note = (
                    f"[Fallback to local model '{local_model_id}' because HF router failed: {api_err}]"
                )
                print(fallback_note)
                text = _local_generate(prompt, model_id=local_model_id, temperature=temp)
                text = f"{text}\n\n{fallback_note}"
            else:
                raise
        return {"messages": messages + [AIMessage(content=text)]}

    graph = StateGraph(ChatState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    app = graph.compile()

    # Wrap to mirror the previous agent_executor interface for Gradio.
    def invoke(payload):
        incoming_messages = payload.get("messages", [])
        initial_state: ChatState = {"messages": incoming_messages, "context": ""}
        return app.invoke(initial_state)

    return SimpleNamespace(invoke=invoke)
