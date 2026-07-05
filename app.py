import streamlit as st

from rag_pipeline import answer_question, build_vectorstore

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Cotton Leaf Disease RAG",
    page_icon="🌱",
    layout="wide",
)

# -----------------------------
# Load Secrets (Streamlit Cloud)
# -----------------------------
GROQ_API_KEY = None
GROQ_MODEL = "llama-3.1-8b-instant"

try:
    GROQ_API_KEY = st.secrets["groq"]["GROQ_API_KEY"]
    GROQ_MODEL = st.secrets["groq"].get("GROQ_MODEL", GROQ_MODEL)
except Exception:
    pass

# -----------------------------
# UI Header
# -----------------------------
st.title("🌿 Cotton Leaf Disease RAG Assistant")
st.write("Ask questions about cotton leaf diseases, causes, and prevention.")

# -----------------------------
# Sidebar Settings
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider("Top K context chunks", 1, 8, 4)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    if st.button("🔄 Rebuild Index"):
        with st.spinner("Building vector store..."):
            build_vectorstore()
        st.success("✅ Index rebuilt successfully!")

# -----------------------------
# API Key Check
# -----------------------------
if not GROQ_API_KEY:
    st.warning("⚠️ GROQ_API_KEY not set. Please add it in Streamlit Secrets.")
    st.stop()  # stop execution if no API key

# -----------------------------
# Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            st.caption("📚 Sources: " + " | ".join(message["sources"]))

# -----------------------------
# Chat Input
# -----------------------------
prompt = st.chat_input("💬 Type your question in Bangla or English...")

if prompt:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    answer = ""
    sources = []

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            try:
                answer, sources = answer_question(
                    prompt,
                    k=top_k,
                    temperature=temperature,
                )
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

        if answer:
            st.markdown(answer)
            if sources:
                st.caption("📚 Sources: " + " | ".join(sources))

    # Save assistant message
    if answer:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )
