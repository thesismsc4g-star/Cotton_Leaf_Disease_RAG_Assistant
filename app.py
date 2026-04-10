import os

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import answer_question, build_vectorstore

load_dotenv()

st.set_page_config(
    page_title="Cotton Leaf Disease RAG",
    page_icon=":seedling:",
    layout="wide",
)

st.title("Cotton Leaf Disease RAG Assistant")
st.write("Ask questions about cotton leaf diseases, causes, and prevention.")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top K context chunks", 1, 8, 4)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    if st.button("Rebuild index"):
        with st.spinner("Building vector store..."):
            build_vectorstore()
        st.success("Index rebuilt.")

if not os.getenv("GROQ_API_KEY"):
    st.warning("Set GROQ_API_KEY in .env to enable answers.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            st.caption("Sources: " + " | ".join(message["sources"]))

prompt = st.chat_input("Type your question in Bangla or English...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    answer = ""
    sources = []
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = answer_question(
                    prompt,
                    k=top_k,
                    temperature=temperature,
                )
            except Exception as exc:
                st.error(str(exc))

        if answer:
            st.markdown(answer)
            if sources:
                st.caption("Sources: " + " | ".join(sources))

    if answer:
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
