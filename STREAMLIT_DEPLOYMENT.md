# Streamlit Hosted Portfolio Deployment

Use Streamlit Community Cloud for the portfolio version of this project. It
matches the actual application architecture: Streamlit UI, LangChain document
processing, OpenAI embeddings, FAISS retrieval, LangGraph workflow, and MCP
export tools.

## Why This Is The Recommended Hosted App

The project needs an interface that can upload files, build a FAISS index, store
the index during the app session, run chat, and expose review/export controls.
Streamlit supports that workflow directly.

Vercel is useful for static frontends and serverless APIs, but the current RAG
workflow needs stateful document/index handling. A Vercel production version
would be better after replacing local FAISS persistence with hosted vector
storage.

## Deploy Steps

1. Go to Streamlit Community Cloud.
2. Create a new app from GitHub.
3. Select repository `Manan-Jain-22/AI_chatbot_with_RAG`.
4. Select branch `main`.
5. Set main file path to `app/streamlit_app.py`.
6. Use Python `3.11` if the UI asks.
7. Add secrets.
8. Deploy.

## Required Secrets

Add these in the Streamlit Cloud app settings:

```toml
OPENAI_API_KEY = "your-openai-api-key"
CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
```

Optional Notion export secrets:

```toml
NOTION_API_KEY = "your-notion-integration-secret"
NOTION_DATABASE_ID = "your-notion-database-id"
NOTION_API_VERSION = "2022-06-28"
```

Do not commit `.streamlit/secrets.toml`. It is intentionally ignored.

## Portfolio Demo Flow

1. Open the hosted Streamlit URL.
2. Confirm the demo notes appear in the sidebar under `View documents`.
3. Click `Build / Rebuild Index`.
4. Ask one of the built-in demo questions:
   - `How should I solve an overdetermined system, and why is QR better than normal equations?`
   - `When should I use conjugate gradient instead of LU factorization?`
   - `What is the difference between Jacobi, Gauss-Seidel, and SOR?`
5. Show the sources under the answer.
6. Explain that MCP export is separated from retrieval because export is an
   external side effect and should happen only after review.

## Interview Positioning

Say this:

```text
I deployed the complete application on Streamlit because my project is not only
a chat endpoint. It includes document upload, chunking, embedding, FAISS index
rebuilding, LangGraph-based retrieval orchestration, answer review, and MCP
export. Streamlit let me host that full workflow as one portfolio app. If I were
turning it into a multi-user production system, I would keep the same LangGraph
RAG backend but move the vector index from local FAISS to hosted vector storage
such as Pinecone, Weaviate, or pgvector.
```
