# AI RAG Chatbot

RAG chatbot built with LangChain, FAISS, OpenAI embeddings, and a LangGraph multi-step workflow. It loads documents, chunks them for semantic search, persists a FAISS index, rewrites questions, retrieves relevant context, and generates cited answers.

## Features

- Load `.pdf`, `.txt`, and `.md` files from `data/`
- Split 100+ documents into retrieval-friendly chunks
- Store semantic embeddings in a local FAISS index
- Use a LangGraph workflow for query rewriting, MCP tool calls, retrieval, and response generation
- Review generated answers before exporting approved responses
- Connect the RAG agent to WhatsApp through a WhatsApp Cloud API MCP tool and webhook
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

## MCP Tool Server

The project includes a local MCP server with tools for query preparation, final response formatting, approved answer export, and WhatsApp Cloud API message sending:

```bash
python -m src.mcp_server
```

LangGraph loads these tools through `langchain-mcp-adapters` when available, with same-schema local fallbacks for restricted environments.

## WhatsApp Access

Streamlit is the full UI for uploading documents and rebuilding the FAISS index. WhatsApp is the lightweight chat channel for asking questions after the index is ready.

Add these values to `.env` from your Meta WhatsApp Cloud API app:

```bash
WHATSAPP_ACCESS_TOKEN=...
WHATSAPP_PHONE_NUMBER_ID=...
WHATSAPP_VERIFY_TOKEN=...
WHATSAPP_API_VERSION=v20.0
```

Run the webhook server:

```bash
uvicorn src.whatsapp_webhook:app --host 0.0.0.0 --port 8000
```

Expose it with a tunnel such as ngrok during development:

```bash
ngrok http 8000
```

Use the public URL as your Meta webhook callback:

```text
https://your-ngrok-domain/webhook/whatsapp
```

The webhook verifies Meta's challenge token on `GET /webhook/whatsapp`, receives user messages on `POST /webhook/whatsapp`, runs the LangGraph RAG workflow, and replies through the MCP-backed WhatsApp send tool.

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
