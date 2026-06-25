# Computational Linear Algebra RAG Study Assistant

RAG study assistant built with LangChain, OpenAI, FAISS, Streamlit, LangGraph, and MCP. The app is focused on Computational Linear Algebra course material rather than generic PDFs. It retrieves conceptually relevant notes for questions about direct methods, sparse systems, Jacobi, Gauss-Seidel, SOR, conjugate gradient, QR, SVD, least squares, and eigenvalue methods.

## Features

- Load `.pdf`, `.txt`, and `.md` Computational Linear Algebra notes from `data/`
- Add metadata for source file, page, inferred topic, and section heading
- Use section-aware chunking so formulas stay near their surrounding explanation
- Store OpenAI embeddings in a local FAISS index for semantic retrieval
- Use LangGraph for query classification, query rewriting, retrieval, context selection, and answer generation
- Evaluate retrieval with a golden set using precision@k and recall@k
- Export finalized answers as Markdown or Notion study cards through MCP tools
- Use Streamlit for uploads, indexing, chat, review, and confirmed export

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your OpenAI API key to `.env`.

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
