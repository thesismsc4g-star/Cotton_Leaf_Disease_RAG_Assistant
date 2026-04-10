from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

BASE_DIR = Path(__file__).resolve().parent
LOCAL_NOTE = BASE_DIR / "Cotton Leaf Disease Management Note.txt"
DATA_DIR = BASE_DIR / "data"
EXTRA_DIR = DATA_DIR / "extra"
SOURCES_JSON = DATA_DIR / "sources.json"
PERSIST_DIR = BASE_DIR / "vectorstore"
COLLECTION_NAME = "cotton_disease"

DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

SYSTEM_PROMPT = (
    "You are an agronomy assistant for cotton leaf diseases. "
    "The user may ask in Bangla or English, but you must answer in English "
    "using clear, short, actionable steps. "
    "Use only the provided context. If the answer is not in the context, "
    "say you do not know and ask for more detail."
)


def load_urls() -> List[str]:
    if not SOURCES_JSON.exists():
        return []
    with SOURCES_JSON.open("r", encoding="utf-8") as file:
        data = json.load(file)
    urls = data.get("urls", [])
    return [url for url in urls if isinstance(url, str) and url.strip()]


def load_local_documents() -> List[Document]:
    documents: List[Document] = []
    if LOCAL_NOTE.exists():
        documents.extend(TextLoader(str(LOCAL_NOTE), encoding="utf-8").load())
    return documents


def load_extra_documents() -> List[Document]:
    documents: List[Document] = []
    if not EXTRA_DIR.exists():
        return documents

    for path in EXTRA_DIR.glob("*.txt"):
        documents.extend(TextLoader(str(path), encoding="utf-8").load())
    for path in EXTRA_DIR.glob("*.md"):
        documents.extend(TextLoader(str(path), encoding="utf-8").load())
    for path in EXTRA_DIR.glob("*.pdf"):
        documents.extend(load_pdf(path))

    return documents


def load_pdf(path: Path) -> List[Document]:
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except Exception:
        return []

    loader = PyPDFLoader(str(path))
    return loader.load()


def load_web_documents(urls: List[str]) -> List[Document]:
    if not urls:
        return []
    loader = WebBaseLoader(urls)
    return loader.load()


def load_documents() -> List[Document]:
    documents = []
    documents.extend(load_local_documents())
    documents.extend(load_extra_documents())
    documents.extend(load_web_documents(load_urls()))
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )
    return splitter.split_documents(documents)


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)


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


def get_llm(temperature: float = 0.2) -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set.")
    return ChatGroq(
        api_key=api_key,
        model=DEFAULT_GROQ_MODEL,
        temperature=temperature,
    )


def answer_question(
    question: str,
    k: int = 4,
    temperature: float = 0.2,
) -> Tuple[str, List[str]]:
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    try:
        docs = retriever.invoke(question)
    except Exception:
        docs = vectorstore.similarity_search(question, k=k)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
    )

    chain = prompt | get_llm(temperature=temperature) | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context})

    sources: List[str] = []
    for doc in docs:
        source = doc.metadata.get("source")
        if source:
            sources.append(source)

    return answer, sorted(set(sources))
