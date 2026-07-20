# Computational Linear Algebra RAG Study Assistant

RAG study assistant built with LangChain, Gemini/OpenAI, FAISS, Streamlit, LangGraph, and MCP. The app is focused on Computational Linear Algebra course material rather than generic PDFs. It retrieves conceptually relevant notes for questions about direct methods, sparse systems, Jacobi, Gauss-Seidel, SOR, conjugate gradient, QR, SVD, least squares, and eigenvalue methods.

## Primary Application

The single complete application is the Streamlit app in `app/streamlit_app.py`.
It handles the full project workflow in one interface:

1. upload course documents
2. build or rebuild the FAISS vector index
3. ask document-grounded questions
4. review the generated answer
5. export approved answers through MCP tools

This is the version to demo in an interview because it shows the actual RAG
pipeline end to end. The Vercel files are only a lightweight deployment surface
for later experimentation; they do not replace the Streamlit app unless the
vector index is moved to persistent hosted storage.

## Features

- Load `.pdf`, `.txt`, and `.md` Computational Linear Algebra notes from `data/`
- Add metadata for source file, page, inferred topic, and section heading
- Use section-aware chunking so formulas stay near their surrounding explanation
- Store provider embeddings in a local FAISS index for semantic retrieval
- Use LangGraph for query classification, query rewriting, retrieval, context selection, and answer generation
- Evaluate retrieval with a golden set using precision@k and recall@k
- Export finalized answers as Markdown or Notion study cards through MCP tools
- Use Streamlit as the main application for uploads, indexing, chat, review, and confirmed export

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Gemini is the default live provider:

```bash
LLM_PROVIDER=gemini
GOOGLE_API_KEY=...
GEMINI_CHAT_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
```

OpenAI remains available only as an optional provider:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=...
CHAT_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

For Notion export, create a Notion integration and a study-card database with these properties:

- `Question`: title
- `Answer`: rich_text
- `Topic`: select
- `Tags`: multi_select
- `Sources`: rich_text

Then add:

```bash
NOTION_API_KEY=...
NOTION_DATABASE_ID=...
```

## Add Course Documents

Create a `data/` folder and add supported course files:

```bash
mkdir -p data
```

Supported file types: `.pdf`, `.txt`, `.md`.

## Build The FAISS Index

```bash
python -m src.vector_store
```

The index is saved to `index/faiss` by default.

## Run The Study Assistant

```bash
streamlit run app/streamlit_app.py
```

You can also ask a question from the terminal:

```bash
python -m src.rag_chain
```

## Hosted Portfolio App

The recommended hosted deployment is Streamlit Community Cloud because the
complete application is a Streamlit app.

Deployment settings:

```text
Repository: Manan-Jain-22/AI_chatbot_with_RAG
Branch: main
Main file path: app/streamlit_app.py
Python version: 3.11
```

For a cost-safe public portfolio demo, add this secret:

```toml
PUBLIC_DEMO_MODE = "true"
```

The repo includes `data/demo_computational_linear_algebra_notes.md`, so the
hosted app has safe demo content. In public demo mode, the sidebar demo
questions work without external model API calls.

For a private live RAG deployment, use:

```toml
PUBLIC_DEMO_MODE = "false"
LLM_PROVIDER = "gemini"
GOOGLE_API_KEY = "..."
GEMINI_CHAT_MODEL = "gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
```

Then upload documents and click **Build / Rebuild Index**.

See `STREAMLIT_DEPLOYMENT.md` for the full hosted deployment checklist.

## Vercel Deployment Note

This repo includes a lightweight Vercel-ready web/API surface:

- `public/index.html`: Vercel static browser chat UI
- `api/health.py`: serverless health endpoint
- `api/chat.py`: serverless chat endpoint that calls the same LangGraph RAG backend
- `vercel.json`: Vercel routing/function configuration

Use Streamlit for the full application. Vercel serverless functions are stateless,
so a production Vercel deployment should either include a sanitized prebuilt
FAISS index or move vectors to a hosted vector database such as Pinecone,
Weaviate, or pgvector.

See `VERCEL_DEPLOYMENT.md` for deployment notes.

## LangGraph Workflow

The graph separates the RAG workflow into explicit steps:

```text
classify_query
-> rewrite_query
-> retrieve
-> select_context
-> generate_answer
```

MCP is intentionally kept separate from retrieval. It is used after a response is finalized, for controlled external export to Markdown or Notion.

## MCP Tool Server

The project includes a local MCP server with export tools:

```bash
python -m src.mcp_server
```

Tools:

- `export_answer_to_markdown`
- `save_study_card_to_notion`

## Retrieval Evaluation

The golden set lives at `eval/golden_questions.csv` and includes representative student-style questions with expected topic/section labels.

Run:

```bash
python -m src.evaluation
```

The script writes precision@k and recall@k results to `eval/results.csv`.
