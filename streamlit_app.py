import streamlit as st
from multiprocessing import Queue, set_start_method
from datetime import datetime
import time
import re
import html as html_lib
from mcp_agent import start_mcp 

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
K_RETRIEVE = 50
K_RERANK = 10
OLLAMA_MODEL = "llama3.2:1b"

def _log_ui(msg: str):
    print(f"[{datetime.now().isoformat()}] [UI] {msg}", flush=True)

def sanitize_text(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text_unesc = html_lib.unescape(text)
    text_no_script = re.sub(r'(?is)<script.*?>.*?</script>', '', text_unesc)
    text_no_style = re.sub(r'(?is)<style.*?>.*?</style>', '', text_no_script)
    cleaned = re.sub(r'<[^>]+>', '', text_no_style).strip()
    return cleaned if cleaned else text.strip()

def sanitize_history():
    if "messages" not in st.session_state:
        return
    new_msgs = []
    for m in st.session_state.messages:
        role = m.get("role", "assistant")
        raw = m.get("text", "")
        cleaned = sanitize_text(raw)
        if not cleaned:
            continue
        if re.match(r'^</?\w+>$', cleaned.strip()):
            continue
        new_msg = {"role": role, "text": cleaned}
        if "sources" in m:
            new_msg["sources"] = m["sources"]
        new_msgs.append(new_msg)
    st.session_state.messages = new_msgs

def init_session_state():
    if "queues" not in st.session_state:
        st.session_state.queues = {
            "ing_in": Queue(), "ing_out": Queue(),
            "ret_in": Queue(), "ret_out": Queue(),
            "llm_in": Queue(), "llm_out": Queue()
        }

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "text": "Hello, Please upload documents in supported documents.",
            "sources": []
        })
    else:
        sanitize_history()

    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    if not st.session_state.get("mcp_started", False):
        try:
            _log_ui("Auto-starting MCP broker and agents...")
            broker, queues = start_mcp(
                embedding_model=EMBEDDING_MODEL,
                K_RETRIEVE=K_RETRIEVE,
                K_RERANK=K_RERANK,
                llm_model=OLLAMA_MODEL,
            )
            st.session_state.mcp_broker = broker
            st.session_state.queues = queues
            st.session_state.mcp_started = True
            _log_ui("MCP started and queues wired to session_state.queues")
        except Exception as e:
            _log_ui(f"Failed to start MCP: {e}")
            raise

def append_user(text: str):
    st.session_state.messages.append({"role": "user", "text": sanitize_text(text)})

def append_assistant(text: str, sources=None):
    st.session_state.messages.append({"role": "assistant", "text": sanitize_text(text), "sources": sources or []})

# ---------- UI ----------
def render_upload_ui(mcp_broker):
    st.header("Upload Files")
    uploaded = st.file_uploader("Choose files", accept_multiple_files=True, type=["pdf", "pptx", "docx", "csv", "txt", "md"])
    if st.button("Ingest files"):
        if not uploaded:
            st.warning("Select files first.")
        else:
            files_payload = []
            for f in uploaded:
                raw = f.read()
                files_payload.append((f.name, raw))
            ts = datetime.now().isoformat()
            _log_ui(f"Ingest pressed at {ts} — sending {len(files_payload)} files to MCP Broker")
            with st.spinner("Parsing uploaded files (indexing runs in background)..."):
                resp = mcp_broker.upload_files(files_payload)
                results = resp.get("payload", {}).get("results", [])
                total_chunks = sum(r.get("num_chunks", 0) for r in results)
                st.success(f"Parsed {total_chunks} chunks from {len(results)} files. Indexing may continue in background.")
                

def render_chat_history_only():
    st.header("Chat")
    for msg in st.session_state.messages:
        role = msg.get("role", "assistant")
        text = msg.get("text", "")
        with st.chat_message(role):
            st.markdown(html_lib.escape(text))

def handle_chat_input(mcp_broker):
    user_prompt = st.chat_input("Ask a question")
    if user_prompt:
        append_user(user_prompt)
        _log_ui(f"User prompt sent to MCP Broker: {user_prompt[:200]}")
        with st.spinner("Retrieving and generating answer..."):
            resp = mcp_broker.ask_query(user_prompt)
            answer = resp.get("payload", {}).get("answer", "")
            retrieved = resp.get("payload", {}).get("retrieved_context", [])
            if answer is None or not str(answer).strip():
                answer = "[No answer returned]"

            with st.chat_message("assistant"):
                placeholder = st.empty()
                ans_str = str(answer)
                chunk_size = 200
                pos = 0
                while pos < len(ans_str):
                    pos = min(len(ans_str), pos + chunk_size)
                    placeholder.markdown(html_lib.escape(ans_str[:pos]))
                    time.sleep(0.03)
                placeholder.markdown(html_lib.escape(ans_str))

            append_assistant(answer, sources=retrieved)
        st.rerun()


# ---------- Main ----------
def main():
    st.set_page_config(page_title="Agentic RAG Chatbot", layout="wide")
    st.title("Agentic RAG Chatbot using MCP")
    init_session_state()
    mcp_broker = st.session_state.mcp_broker
    left_col, right_col = st.columns([1, 2], gap="large")
    with left_col:
        render_upload_ui(mcp_broker)
    with right_col:
        render_chat_history_only()
    handle_chat_input(mcp_broker)


if __name__ == "__main__":
    main()
