# AI RAG Chatbot

RAG chatbot built with LangChain, FAISS, OpenAI embeddings, and a LangGraph multi-step workflow. It loads documents, chunks them for semantic search, persists a FAISS index, rewrites questions, retrieves relevant context, and generates cited answers.

## Features

- Load `.pdf`, `.txt`, and `.md` files from `data/`
- Split 100+ documents into retrieval-friendly chunks
- Store semantic embeddings in a local FAISS index
- Use a LangGraph workflow for query rewriting, optional tool execution, retrieval, and response generation
- Run through Streamlit or a terminal script

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your OpenAI API key to `.env`.

## Add Documents

Create a `data/` folder and add supported files:

```bash
mkdir -p data
```

Supported file types: `.pdf`, `.txt`, `.md`.

## Build The FAISS Index

```bash
python -m src.vector_store
```

The index is saved to `index/faiss` by default.

## Run The Chatbot

```bash
streamlit run app/streamlit_app.py
```

You can also ask a question from the terminal:

```bash
python -m src.rag_chain
```

## Evaluation

Create `eval/questions.csv`:

```csv
question
What is the main topic of the documents?
```

Then run:

```bash
python -m src.evaluation
```
