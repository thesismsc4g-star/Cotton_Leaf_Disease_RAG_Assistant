from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import streamlit as st

from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# =========================
# PATH CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
LOCAL_NOTE = BASE_DIR / "Cotton Leaf Disease Management Note.txt"
DATA_DIR = BASE_DIR / "data"
EXTRA_DIR = DATA_DIR / "extra"
SOURCES_JSON = DATA_DIR / "sources.json"
PERSIST_DIR = BASE_DIR / "vectorstore"
COLLECTION_NAME = "cotton_disease"

# =========================
# CONFIG (SECRETS + ENV)
# =========================
def get_config():
    try:
        return {
            "embedding": st.secrets["embedding"]["EMBEDDING_MODEL"],
            "groq_model": st.secrets["groq"]["GROQ_MODEL"],
            "api_key": st.secrets["groq"]["GROQ_API_KEY"],
        }
    except Exception:
        return {
            "embedding": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            "groq_model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            "api_key": os.getenv("GROQ_API_KEY", ""),
        }

# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = (
    "You are a specialized agronomy assistant focused on cotton leaf diseases and their management.\n\n"

    "Your responsibilities:\n"
    "- Identify cotton leaf diseases from symptoms (spots, discoloration, wilting, abnormal growth).\n"
    "- Recognize specific disease names such as:\n"
    "  Alternaria Leaf Spot, Bacterial Blight, Angular Leaf Spot, Leaf Spot,\n"
    "  Fusarium, Fusarium Wilt, Verticillium, Verticillium Wilt,\n"
    "  Leaf Curl, Leaf Curls, Leaf Hopper, Jassids,\n"
    "  Leaf Blight, Herbicide Growth Damage, Leaf Reddening,\n"
    "  Leaf Variegation, Healthy Leaf, Cotton Rust, Anthracnose.\n"
    "- Explain causes (fungal, bacterial, viral, pest-related, or chemical damage).\n"
    "- Provide short, clear, actionable treatment steps.\n"
    "- Suggest prevention methods and proper farming practices.\n\n"

    "STRICT RULES:\n"
    "- Use ONLY the provided context.\n"
    "- If the answer is not in the context, say:\n"
    "  'I do not know based on the available data. Please provide more details.'\n"
    "- If the question is NOT related to cotton leaf diseases OR the above listed diseases, say:\n"
    "  'I am a specialized assistant for cotton leaf diseases and cannot answer this question.'\n\n"

    "Language rules:\n"
    "- User may use Bangla or English.\n"
    "- ALWAYS respond in English.\n"
    "- Keep answers short, clear, and practical.\n"
)

keywords = [
    "cotton", "leaf", "disease", "plant", "crop",
    "pest", "fungus", "yellow", "spot", "blight",

    "alternaria", "alternaria leaf spot",
    "bacterial blight",
    "angular leaf spot",
    "leaf spot",
    "fusarium", "fusarium wilt",
    "verticillium", "verticillium wilt",
    "leaf curl", "leaf curls",
    "leaf hopper", "jassids",
    "leaf blight",
    "herbicide damage", "growth damage",
    "leaf reddening",
    "leaf variegation",
    "healthy leaf",
    "leaf diseases",
    "cotton rust",
    "anthracnose"
]

# =========================
# FALLBACK MESSAGE
# =========================
FALLBACK_MESSAGE = (
    "I'm a specialized assistant focused on cotton leaf diseases and their management.\n\n"

    "My purpose is to help farmers, researchers, and students better understand issues affecting cotton plant health. "
    "I can assist you with identifying different types of cotton leaf diseases based on symptoms such as spots, "
    "discoloration, wilting, or abnormal growth patterns. I can also explain the underlying causes, including fungal, "
    "bacterial, viral infections, and pest-related damage.\n\n"

    "In addition, I can provide practical guidance on disease prevention, control measures, and recommended treatment "
    "strategies, including proper use of fertilizers, pesticides, and cultivation practices to maintain healthy cotton crops.\n\n"

    "At the moment, I’m not able to assist with queries outside this domain. Please feel free to ask any question related "
    "to cotton leaf diseases or cotton crop management, and I’ll be happy to help."
)

# =========================
# DATA LOADERS
# =========================
def load_urls() -> List[str]:
    if not SOURCES_JSON.exists():
        return []
    with SOURCES_JSON.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return [url for url in data.get("urls", []) if isinstance(url, str) and url.strip()]


def load_local_documents() -> List[Document]:
    docs: List[Document] = []
    if LOCAL_NOTE.exists():
        docs.extend(TextLoader(str(LOCAL_NOTE), encoding="utf-8").load())
    return docs


def load_extra_documents() -> List[Document]:
    docs: List[Document] = []
    if not EXTRA_DIR.exists():
        return docs

    for path in EXTRA_DIR.glob("*.txt"):
        docs.extend(TextLoader(str(path), encoding="utf-8").load())

    for path in EXTRA_DIR.glob("*.md"):
        docs.extend(TextLoader(str(path), encoding="utf-8").load())

    for path in EXTRA_DIR.glob("*.pdf"):
        docs.extend(load_pdf(path))

    return docs


def load_pdf(path: Path) -> List[Document]:
    try:
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(str(path)).load()
    except Exception:
        return []


def load_web_documents(urls: List[str]) -> List[Document]:
    return WebBaseLoader(urls).load() if urls else []


def load_documents() -> List[Document]:
    docs = []
    docs.extend(load_local_documents())
    docs.extend(load_extra_documents())
    docs.extend(load_web_documents(load_urls()))
    return docs

# =========================
# TEXT SPLIT
# =========================
def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    return splitter.split_documents(documents)

# =========================
# EMBEDDINGS
# =========================
def get_embeddings() -> HuggingFaceEmbeddings:
    config = get_config()
    return HuggingFaceEmbeddings(model_name=config["embedding"])

# =========================
# VECTORSTORE
# =========================
def build_vectorstore() -> Chroma:
    documents = load_documents()
    if not documents:
        raise ValueError("No documents found. Add data sources and try again.")

    splits = split_documents(documents)
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        splits,
        embeddings,
        persist_directory=str(PERSIST_DIR),
        collection_name=COLLECTION_NAME,
    )
    vectorstore.persist()
    return vectorstore


def get_vectorstore() -> Chroma:
    embeddings = get_embeddings()

    if not PERSIST_DIR.exists() or not any(PERSIST_DIR.iterdir()):
        return build_vectorstore()

    return Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

# =========================
# LLM
# =========================
def get_llm(temperature: float = 0.2) -> ChatGroq:
    config = get_config()

    if not config["api_key"]:
        raise EnvironmentError("GROQ_API_KEY is not set.")

    return ChatGroq(
        api_key=config["api_key"],
        model=config["groq_model"],
        temperature=temperature,
    )

# =========================
# MAIN QA FUNCTION
# =========================
def answer_question(
    question: str,
    k: int = 4,
    temperature: float = 0.2,
) -> Tuple[str, List[str]]:

    q = question.lower()

    # ✅ Domain filter
    keywords = [
        "cotton", "leaf", "disease", "plant", "crop",
        "pest", "fungus", "yellow", "spot", "blight"
    ]

    if not any(word in q for word in keywords):
        return FALLBACK_MESSAGE, []

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    try:
        docs = retriever.invoke(question)
    except Exception:
        docs = vectorstore.similarity_search(question, k=k)

    context = "\n\n".join(doc.page_content for doc in docs)

    # ❌ No context
    if not context.strip():
        return FALLBACK_MESSAGE, []

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
    )

    chain = prompt | get_llm(temperature) | StrOutputParser()
    answer = chain.invoke({
        "question": question,
        "context": context
    }).strip()

    sources = list({
        doc.metadata.get("source")
        for doc in docs
        if doc.metadata.get("source")
    })

    return answer, sources
