# Cotton Leaf Disease RAG Assistant

This project is a Retrieval-Augmented Generation (RAG) chatbot for cotton leaf disease management. It uses a local knowledge note plus optional open-source web sources, builds a vector store, and generates grounded answers with Groq (LLM) via LangChain. Users can ask in Bangla or English, but answers are returned in English with short, actionable steps.

## Features
- RAG-based answers from your note file and optional web sources.
- English responses with short, actionable steps.
- Streamlit chat UI with source citations.
- Simple indexing workflow and fast re-build.

## How it works
1. Load local notes and extra documents (txt, md, pdf).
2. Optionally fetch open-source URLs from data/sources.json.
3. Split content into chunks and embed them.
4. Store embeddings in a local Chroma vectorstore.
5. Retrieve relevant chunks and generate answers using Groq.

## Model and workflow (short)
### Which model is used for what
- Embedding: sentence-transformers/all-MiniLM-L6-v2 (env: `EMBEDDING_MODEL`) — converts text into vectors.
- LLM: llama-3.1-8b-instant (env: `GROQ_MODEL`) — generates the final answer from retrieved context.
- RAG model: there is no single RAG model here; RAG is the pipeline that combines retrieval (Chroma) + embeddings + LLM.

### Workflow (short)
1. Load local note + extra files + optional URLs.
2. Chunk the text.
3. Embed chunks and save to the Chroma vectorstore.
4. For a user query, run similarity search and retrieve top-k chunks.
5. Generate the answer with Groq LLM and show sources.

## Tech stack
- UI: Streamlit
- RAG: LangChain + Chroma
- Embeddings: sentence-transformers
- LLM: Groq API

## Project structure
- app.py: Streamlit UI
- rag_pipeline.py: RAG pipeline (load, split, embed, retrieve, generate)
- ingest.py: builds or refreshes vectorstore
- data/sources.json: open-source URLs list
- data/extra/: extra local files (txt, md, pdf)
- Cotton Leaf Disease Management Note.txt: primary knowledge base

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   pip install -r requirements.txt
3. Copy .env.example to .env and set GROQ_API_KEY.
4. Build the vector store:
   python ingest.py
5. Run the app:
   streamlit run app.py

## Run with venv explicitly (Windows)
If you want to be sure Streamlit uses the venv:
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  .venv\Scripts\python.exe -m streamlit run app.py

## Data sources
- Main note file: Cotton Leaf Disease Management Note.txt
- Add extra local files to data/extra (txt, md, pdf)
- Add open-source URLs to data/sources.json and re-run python ingest.py

## Sample questions (English)
1. What are the causes and prevention of Alternaria leaf spot?
2. What are the early symptoms of bacterial blight?
3. What crop rotation is best for Fusarium wilt management?
4. Why do cooler temperatures favor Verticillium wilt?
5. When should I apply fungicide for leaf spot?
6. Why is field scouting important and how often should it be done?
7. How does overhead irrigation increase bacterial blight risk?
8. Give 5 actionable steps to prevent cotton leaf diseases.
9. Which disease is more likely with poor soil drainage?
10. Provide a short checklist for healthy leaf management.

## Notes
- The chatbot answers in English and uses only retrieved context.
- If the answer is not in the context, it will say it does not know.

## Troubleshooting
- Missing module errors: run pip install -r requirements.txt in the same venv used to run Streamlit.
- Protobuf descriptor error: downgrade protobuf in the venv:
  .venv\Scripts\python.exe -m pip install "protobuf>=3.20.3,<4"
