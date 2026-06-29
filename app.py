# """
# app.py

# Streamlit web application for the NileTech Solutions Ltd.
# Policy Retrieval-Augmented Generation (RAG) Assistant.
# """

# import streamlit as st
# from src.rag_chain import PolicyRAG

# # -----------------------------------------------------------------------------
# # Page Configuration
# # -----------------------------------------------------------------------------

# st.set_page_config(
#     page_title="NileTech Policy Assistant",
#     page_icon="📄",
#     layout="wide",
# )

# st.title("📄 NileTech Policy Assistant")
# st.caption(
#     "Ask questions about NileTech Solutions Ltd. policies. "
#     "Responses are generated only from the indexed company policy documents."
# )

# # -----------------------------------------------------------------------------
# # Initialize RAG Pipeline
# # -----------------------------------------------------------------------------

# @st.cache_resource
# def load_rag():
#     return PolicyRAG()


# rag = load_rag()

# # -----------------------------------------------------------------------------
# # Chat History
# # -----------------------------------------------------------------------------

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # -----------------------------------------------------------------------------
# # Chat Input
# # -----------------------------------------------------------------------------

# user_question = st.chat_input("Ask a question about company policies...")

# if user_question:

#     # Display user message
#     st.session_state.messages.append(
#         {
#             "role": "user",
#             "content": user_question,
#         }
#     )

#     with st.chat_message("user"):
#         st.markdown(user_question)

#     # Generate response
#     with st.chat_message("assistant"):
#         with st.spinner("Searching company policies..."):

#             answer = rag.generate(user_question)

#             st.markdown(answer)

#     # Save assistant response
#     st.session_state.messages.append(
#         {
#             "role": "assistant",
#             "content": answer,
#         }
#     )

"""
app.py

Streamlit frontend for the NileTech Solutions Ltd.
Policy Retrieval-Augmented Generation (RAG) Assistant.
"""

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

st.title("📄 NileTech Policy Assistant")
st.caption(
    "Ask questions about NileTech Solutions Ltd. policies. "
    "Responses are generated only from the indexed company policy documents."
)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

st.sidebar.title("System Status")

try:
    health = requests.get("http://127.0.0.1:5000/health", timeout=5)

    if health.status_code == 200:
        st.sidebar.success("🟢 Backend Online")
    else:
        st.sidebar.error("🔴 Backend Error")

except requests.exceptions.RequestException:
    st.sidebar.error("🔴 Backend Offline")

# -----------------------------------------------------------------------------
# Chat History
# -----------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------------------------------------------------------
# Chat Input
# -----------------------------------------------------------------------------

user_question = st.chat_input("Ask a question about company policies...")

if user_question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):

        with st.spinner("Searching company policies..."):

            try:

                response = requests.post(
                    "http://127.0.0.1:5000/chat",
                    json={"question": user_question},
                    timeout=120,
                )

                response.raise_for_status()

                result = response.json()

                answer = result["answer"]

                st.markdown(answer)

                if result.get("sources"):

                    st.markdown("---")
                    st.subheader("📚 Sources")

                    for source in result["sources"]:

                        with st.expander(source["source"]):

                            st.write(source["snippet"])

            except Exception as e:

                answer = f"❌ Error communicating with backend:\n\n{e}"

                st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )