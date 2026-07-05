import streamlit as st

from rag_pipeline import answer_question, build_vectorstore

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Cotton Leaf Disease RAG",
    page_icon="🌿",
    layout="wide",
)

# =====================================
# LOAD CONFIG
# =====================================

try:
    GROQ_API_KEY = st.secrets["groq"]["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = None

# =====================================
# HEADER
# =====================================

st.title("🌿 Cotton Leaf Disease Assistant")

st.markdown(
    """
Ask questions about:

- 🌱 Cotton leaf diseases
- 🦠 Disease symptoms
- 🧪 Causes
- 💊 Treatment
- 🛡️ Prevention

**You can ask in Bangla or English.**
"""
)

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.header("⚙️ Settings")

    top_k = st.slider(
        "Top K Retrieved Chunks",
        min_value=1,
        max_value=8,
        value=4,
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
    )

    st.divider()

    if st.button("🔄 Rebuild Knowledge Base"):

        with st.spinner("Building Vector Database..."):

            build_vectorstore()

        st.success("Vector Database Rebuilt Successfully!")

# =====================================
# API CHECK
# =====================================

if not GROQ_API_KEY:

    st.error("GROQ_API_KEY not found in Streamlit Secrets.")

    st.stop()

# =====================================
# CHAT HISTORY
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            st.caption(
                "📚 Sources: "
                + " | ".join(message["sources"])
            )

# =====================================
# USER INPUT
# =====================================

question = st.chat_input(
    "💬 Ask your question in Bangla or English..."
)

# =====================================
# ANSWER
# =====================================

if question:

    # Show user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # Generate Answer

    with st.chat_message("assistant"):

        with st.spinner("🤖 Thinking..."):

            try:

                answer, sources = answer_question(
                    question=question,
                    k=top_k,
                    temperature=temperature,
                )

            except Exception as e:

                answer = f"❌ Error:\n\n{str(e)}"

                sources = []

        st.markdown(answer)

        if sources:

            st.caption(
                "📚 Sources: "
                + " | ".join(sources)
            )

    # Save Assistant Message

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
