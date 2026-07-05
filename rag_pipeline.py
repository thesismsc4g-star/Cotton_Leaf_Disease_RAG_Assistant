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
    "You are a specialized agronomy assistant focused ONLY on cotton leaf diseases and their management.\n\n"

    "Your responsibilities:\n"
    "- Identify cotton leaf diseases from symptoms such as spots, discoloration, wilting, curling, reddening, abnormal growth, or other visible leaf symptoms.\n"
    "- Recognize diseases including:\n"
    "  Alternaria Leaf Spot,\n"
    "  Bacterial Blight,\n"
    "  Angular Leaf Spot,\n"
    "  Leaf Spot,\n"
    "  Fusarium,\n"
    "  Fusarium Wilt,\n"
    "  Verticillium,\n"
    "  Verticillium Wilt,\n"
    "  Leaf Curl,\n"
    "  Leaf Curls,\n"
    "  Leaf Hopper,\n"
    "  Jassids,\n"
    "  Leaf Blight,\n"
    "  Herbicide Growth Damage,\n"
    "  Leaf Reddening,\n"
    "  Leaf Variegation,\n"
    "  Healthy Leaf,\n"
    "  Cotton Rust,\n"
    "  Anthracnose.\n"

    "- Explain the possible causes (fungal, bacterial, viral, insect/pest-related, nutrient-related, or herbicide damage).\n"
    "- Suggest short, practical treatment recommendations.\n"
    "- Recommend disease prevention and good farming practices.\n\n"

    "STRICT RULES:\n"
    "- Use ONLY the provided context to answer.\n"
    "- Do NOT use outside knowledge.\n"
    "- Do NOT guess or make up information.\n"
    "- If the answer is not available in the provided context, reply:\n"
    "  'I do not know based on the available data. Please provide more details.'\n"
    "- If the user's question is NOT related to cotton, cotton leaf diseases, cotton pests, cotton crop management, or the diseases listed above, reply:\n"
    "  'I am a specialized assistant for cotton leaf diseases and cannot answer this question.'\n\n"

    "Language Rules:\n"
    "- Detect the language of the user's query before answering.\n"
    "- ALWAYS reply in the SAME language as the user's query.\n"
    "- If the user's query is in Bangla, respond entirely in Bangla.\n"
    "- If the user's query is in English, respond entirely in English.\n"
    "- If the user's query contains both Bangla and English, respond naturally in the same mixed language style.\n"
    "- Never change or translate the user's preferred language.\n"
    "- Keep scientific disease names and technical terms (e.g., Alternaria Leaf Spot, Fusarium Wilt, Verticillium Wilt, Anthracnose) in English.\n"
    "- Keep answers short, clear, practical, and easy to understand."
)

# =========================
# KEYWORDS
# =========================

keywords = [
    # Crop
    "cotton",
    "cotton leaf",
    "cotton plant",
    "cotton crop",

    # General
    "leaf",
    "disease",
    "leaf disease",
    "crop disease",
    "plant disease",
    "plant",
    "crop",

    # Symptoms
    "spot",
    "spots",
    "leaf spot",
    "yellow",
    "yellowing",
    "brown",
    "black spot",
    "reddening",
    "curl",
    "curling",
    "wilt",
    "wilting",
    "blight",
    "discoloration",
    "chlorosis",
    "necrosis",

    # Diseases
    "alternaria",
    "alternaria leaf spot",
    "bacterial blight",
    "angular leaf spot",
    "fusarium",
    "fusarium wilt",
    "verticillium",
    "verticillium wilt",
    "leaf curl",
    "leaf curls",
    "leaf blight",
    "cotton rust",
    "anthracnose",
    "leaf reddening",
    "leaf variegation",
    "healthy leaf",

    # Pests
    "leaf hopper",
    "leafhopper",
    "jassid",
    "jassids",
    "pest",
    "insect",

    # Chemical Damage
    "herbicide damage",
    "growth damage",

    # Management
    "treatment",
    "control",
    "management",
    "prevention",
    "fungicide",
    "pesticide",
    "fertilizer",

    # Bangla Keywords
    "তুলা",
    "কার্পাস",
    "তুলা গাছ",
    "পাতা",
    "পাতার রোগ",
    "রোগ",
    "দাগ",
    "হলুদ",
    "কালো দাগ",
    "লিফ স্পট",
    "ব্লাইট",
    "উইল্ট",
    "পাতা কুঁকড়ানো",
    "পাতা লাল হওয়া",
    "ছত্রাক",
    "ব্যাকটেরিয়া",
    "ভাইরাস",
    "পোকা",
    "জ্যাসিড",
    "চিকিৎসা",
    "প্রতিকার",
    "ব্যবস্থাপনা",
    "প্রতিরোধ"
]


# =========================
# FALLBACK MESSAGES
# =========================

BANGLA_FALLBACK = (
    "আমি শুধুমাত্র তুলা গাছের পাতার রোগ এবং তার ব্যবস্থাপনা সম্পর্কিত প্রশ্নের উত্তর দিতে পারি।\n\n"
    "অনুগ্রহ করে তুলা পাতার রোগ, লক্ষণ, চিকিৎসা, প্রতিরোধ, পোকামাকড় বা তুলা চাষ ব্যবস্থাপনা সম্পর্কিত প্রশ্ন করুন।"
)

ENGLISH_FALLBACK = (
 "I'm a specialized assistant focused on cotton leaf diseases and their management.\n\n"

    "My purpose is to help farmers, researchers, and students better understand issues affecting cotton plant health. \n\n"
    "I can assist you with identifying different types of cotton leaf diseases based on symptoms such as spots, \n\n"
    "discoloration, wilting, or abnormal growth patterns. I can also explain the underlying causes, including fungal, \n\n"
    "bacterial, viral infections, and pest-related damage.\n\n"
     "In addition, I can provide practical guidance on disease prevention, control measures, and recommended treatment \n\n"
    "strategies, including proper use of fertilizers, pesticides, and cultivation practices to maintain healthy cotton crops.\n\n"
    "At the moment, I’m not able to assist with queries outside this domain. Please feel free to ask any question related \n\n"
    "to cotton leaf diseases or cotton crop management, and I’ll be happy to help.\n\n"

)


# =========================
# LANGUAGE DETECTION
# =========================

def is_bangla(text: str) -> bool:
    """
    Returns True if the text contains Bangla characters.
    """
    return any('\u0980' <= ch <= '\u09FF' for ch in text)


# =========================
# GET FALLBACK MESSAGE
# =========================

def get_fallback_message(user_query: str) -> str:
    """
    Returns the fallback message based on the user's language.
    """
    if is_bangla(user_query):
        return BANGLA_FALLBACK
    else:
        return ENGLISH_FALLBACK

# =========================
# DATA LOADERS
# =========================
def load_urls() -> List[str]:
    if not SOURCES_JSON.exists():
        return []

    with SOURCES_JSON.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return [
        url
        for url in data.get("urls", [])
        if isinstance(url, str) and url.strip()
    ]


def load_local_documents() -> List[Document]:
    docs: List[Document] = []

    if LOCAL_NOTE.exists():
        docs.extend(
            TextLoader(
                str(LOCAL_NOTE),
                encoding="utf-8",
            ).load()
        )

    return docs


def load_extra_documents() -> List[Document]:
    docs: List[Document] = []

    if not EXTRA_DIR.exists():
        return docs

    for path in EXTRA_DIR.glob("*.txt"):
        docs.extend(
            TextLoader(
                str(path),
                encoding="utf-8",
            ).load()
        )

    for path in EXTRA_DIR.glob("*.md"):
        docs.extend(
            TextLoader(
                str(path),
                encoding="utf-8",
            ).load()
        )

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
    if not urls:
        return []

    return WebBaseLoader(urls).load()


def load_documents() -> List[Document]:
    docs = []

    docs.extend(load_local_documents())
    docs.extend(load_extra_documents())
    docs.extend(load_web_documents(load_urls()))

    return docs


# =========================
# TEXT SPLITTER
# =========================
def split_documents(documents: List[Document]) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )

    return splitter.split_documents(documents)


# =========================
# EMBEDDINGS
# =========================
def get_embeddings() -> HuggingFaceEmbeddings:

    config = get_config()

    return HuggingFaceEmbeddings(
        model_name=config["embedding"]
    )


# =========================
# VECTOR STORE
# =========================
def build_vectorstore() -> Chroma:

    documents = load_documents()

    if not documents:
        raise ValueError(
            "No documents found. Add data sources first."
        )

    splits = split_documents(documents)

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
        collection_name=COLLECTION_NAME,
    )

    vectorstore.persist()

    return vectorstore


def get_vectorstore() -> Chroma:

    embeddings = get_embeddings()

    if (
        not PERSIST_DIR.exists()
        or not any(PERSIST_DIR.iterdir())
    ):
        return build_vectorstore()

    return Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


# =========================
# LLM
# =========================
def get_llm(
    temperature: float = 0.2,
) -> ChatGroq:

    config = get_config()

    if not config["api_key"]:
        raise EnvironmentError(
            "GROQ_API_KEY is not set."
        )

    return ChatGroq(
        api_key=config["api_key"],
        model=config["groq_model"],
        temperature=temperature,
    )


# =========================
# QUESTION ANSWERING
# =========================
def answer_question(
    question: str,
    k: int = 4,
    temperature: float = 0.2,
) -> Tuple[str, List[str]]:

    q = question.lower()

    # -----------------------
    # Domain Filter
    # -----------------------
    if not any(
        keyword.lower() in q
        for keyword in keywords
    ):
        return get_fallback_message(question), []

    # -----------------------
    # Retrieve Context
    # -----------------------
    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    try:
        docs = retriever.invoke(question)

    except Exception:
        docs = vectorstore.similarity_search(
            question,
            k=k,
        )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    # -----------------------
    # No Context
    # -----------------------
    if not context.strip():
        return get_fallback_message(question), []

    # -----------------------
    # Prompt
    # -----------------------
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Question:\n{question}\n\nContext:\n{context}",
            ),
        ]
    )

    chain = (
        prompt
        | get_llm(temperature)
        | StrOutputParser()
    )

    answer = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    ).strip()

    # -----------------------
    # Sources
    # -----------------------
    sources = list(
        {
            doc.metadata.get("source")
            for doc in docs
            if doc.metadata.get("source")
        }
    )

    return answer, sources
