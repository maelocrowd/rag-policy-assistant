"""
app.py

Streamlit frontend for the NileTech Solutions Ltd.
Policy Retrieval-Augmented Generation (RAG) Assistant.
"""
import os
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="NileTech Policy Assistant",
    page_icon="📄",
    layout="wide",
)
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5000").rstrip("/")
st.title("📄 NileTech Policy Assistant")
st.caption(
    "Ask questions about NileTech Solutions Ltd. policies. "
    "Responses are generated only from the indexed company policy documents."
)

# -----------------------------------------------------------------------------
# Sidebar Status & Control Layout
# -----------------------------------------------------------------------------

st.sidebar.title("System Status")

try:
    # FIX: Corrected target address from 127.0.0.0 to 127.0.0.1 loopback IP
    health = requests.get(f"{BACKEND_URL}/health", timeout=5)
    if health.status_code == 200:
        st.sidebar.success("🟢 Backend Online")
    else:
        st.sidebar.error("🔴 Backend Error")
except requests.exceptions.RequestException:
    st.sidebar.error("🔴 Backend Offline")

st.sidebar.markdown("---")
st.sidebar.title("Controls")

# Quick reset capability for session state clearing
if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# -----------------------------------------------------------------------------
# Chat History Logic and Helper UI Formatters
# -----------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    """Groups and displays source snippets by filename to prevent multi-expander clutter."""
    if not sources:
        return
        
    st.markdown("---")
    st.subheader("📚 Sources")
    
    # Categorize raw flat chunk responses by their parent file
    grouped_sources = {}
    for item in sources:
        src_file = item.get("source", "Unknown Document")
        if src_file not in grouped_sources:
            grouped_sources[src_file] = []
        grouped_sources[src_file].append(item)
        
    # Build unique, cleanly structured expanders per source file
    for idx, (file_name, chunks) in enumerate(grouped_sources.items(), start=1):
        with st.expander(f"📁 {idx}. {file_name} ({len(chunks)} section/s)"):
            for chunk_idx, chunk in enumerate(chunks, start=1):
                section_name = chunk.get("section", "General Section")
                snippet_text = chunk.get("snippet", "").strip()
                
                if chunk_idx > 1:
                    st.write("---")
                    
                # FIX: Restored missing elements and closed out the UI format block cleanly
                st.markdown(f"**📍 Section:** {section_name}")
                st.markdown(f"```text\n{snippet_text}\n```")


# Display historical messages across script re-runs
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            render_sources(message["sources"])

# -----------------------------------------------------------------------------
# Chat Input & Active API Communications Loop
# -----------------------------------------------------------------------------

user_question = st.chat_input("Ask a question about company policies...")

if user_question:

    # 1. Pipeline Input Phase
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # 2. Pipeline Execution/Response Phase
    with st.chat_message("assistant"):
        with st.spinner("Searching company policies..."):
            
            answer = ""
            sources = []
            
            try:
                response = requests.post(f"{BACKEND_URL}/chat", json={"question": user_question}, timeout=120)
                response.raise_for_status()
                result = response.json()
                
                answer = result.get("answer", "No answer text returned.")
                sources = result.get("sources", [])
                
                # Active UI Rendering
                st.markdown(answer)
                render_sources(sources)

            except Exception as e:
                answer = f"❌ Error communicating with backend:\n\n{e}"
                st.error(answer)
                sources = []

    # 3. Commit Frame States
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )
