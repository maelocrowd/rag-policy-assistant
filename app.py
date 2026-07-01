"""
app.py

Streamlit frontend for the NileTech Solutions Ltd.
Policy Retrieval-Augmented Generation (RAG) Assistant.
"""
import os
import re
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
    health = requests.get(f"{BACKEND_URL}/health", timeout=5)
    if health.status_code == 200:
        st.sidebar.success("🟢 Backend Online")
    else:
        st.sidebar.error("🔴 Backend Error")
except requests.exceptions.RequestException:
    st.sidebar.error("🔴 Backend Offline")

st.sidebar.markdown("---")
st.sidebar.title("Controls")

if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# -----------------------------------------------------------------------------
# Chat History Logic and Helper UI Formatters
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


def format_inline_citations(text: str) -> str:
    """
    Finds strict bracketed citations [Section: ..., file.md] and transforms
    them into clean, visually appealing reference badges.
    """
    if not text:
        return text

    # FIXED: Restored clean standard string pipes (|) to resolve matching breaks
    pattern = r"\[Section:\s*(.*?),\s*([A-Za-z0-9_.\-]+\.(?:md\vert{}pdf\vert{}txt\vert{}html))\]"
    
    def badge_replacer(match):
        section_name = match.group(1).strip()
        file_name = match.group(2).strip()
        short_section = (section_name[:15] + "...") if len(section_name) > 18 else section_name
        
        return f'<span style="background-color: #e0f2fe; color: #0369a1; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; font-weight: bold; border: 1px solid #bae6fd; margin-left: 4px;" title="Source: {file_name} (Section: {section_name})">🔖 {file_name} › {short_section}</span>'

    text = re.sub(pattern, badge_replacer, text)
    
    # FIXED: Standalone fallback pattern clean pipe validation
    fallback_pattern = r"\[([A-Za-z0-9_.\-]+\.(?:md\vert{}pdf\vert{}txt\vert{}html))\]"
    text = re.sub(fallback_pattern, r'<span style="background-color: #e0f2fe; color: #0369a1; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; font-weight: bold; border: 1px solid #bae6fd; margin-left: 4px;">🔖 \1</span>', text)
    
    return text


def render_sources(sources):
    """
    Groups source snippets by filename into custom HTML highlighted bluish cards,
    and categorizes nested snippets neatly into unified section buckets if duplicated.
    """
    if not sources:
        return
        
    st.markdown("---")
    st.subheader("📚 Sources")
    
    # Tier 1: Group flat chunk data entries by their target Document File
    grouped_by_doc = {}
    for item in sources:
        src_file = item.get("source", "Unknown Document")
        if src_file not in grouped_by_doc:
            grouped_by_doc[src_file] = []
        grouped_by_doc[src_file].append(item)
        
    # Build expanders per source file using explicit styling overrides
    for idx, (file_name, items) in enumerate(grouped_by_doc.items(), start=1):
        
        # Tier 2: Further group text snippets within this document by their Section Category
        sections_map = {}
        for item in items:
            sec_name = item.get("section", "General Section")
            snippet_text = item.get("snippet", "").strip()
            
            if sec_name not in sections_map:
                sections_map[sec_name] = []
            
            # Simple content deduplication check to prevent repeating identical strings
            if snippet_text not in sections_map[sec_name]:
                sections_map[sec_name].append(snippet_text)
        
        # FIXED: Render the expander header natively, then force-inject the custom blue panel content layout block
        with st.expander(f"📁 {idx}. {file_name} ({len(sections_map)} unique section/s)"):
            
            # Open custom styled blue highlight outer container wrap
            st.markdown(
                '<div style="background-color: #f0f4f8; padding: 15px; border-left: 5px solid #2563eb; border-radius: 4px; margin-top: 5px; margin-bottom: 5px;">', 
                unsafe_allow_html=True
            )
            
            # Display contents organized cleanly by their extracted Policy Category headings
            for s_idx, (section_name, snippets) in enumerate(sections_map.items()):
                if s_idx > 0:
                    st.markdown("<hr style='margin: 12px 0; border-top: 1px dashed #cbd5e1;' />", unsafe_allow_html=True)
                
                st.markdown(f"<p style='color: #1e40af; font-weight: bold; margin-top: 0; margin-bottom: 6px; font-size: 1.05em;'>📍 Section: {section_name}</p>", unsafe_allow_html=True)
                
                for snippet in snippets:
                    st.markdown(f"```text\n{snippet}\n```")
            
            # Close custom blue inner container wrap
            st.markdown('</div>', unsafe_allow_html=True)


# Display historical messages across script re-runs
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(format_inline_citations(message["content"]), unsafe_allow_html=True)
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
                
                # Active UI Rendering with HTML badge parsing
                st.markdown(format_inline_citations(answer), unsafe_allow_html=True)
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
