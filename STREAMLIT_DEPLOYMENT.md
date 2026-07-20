# Streamlit Hosted Portfolio Deployment

Use Streamlit Community Cloud for the portfolio version of this project. It
matches the actual application architecture: Streamlit UI, LangChain document
processing, Gemini/OpenAI embeddings, FAISS retrieval, LangGraph workflow, and MCP
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

## Cost-Safe Public Demo Secrets

For the public portfolio app, use this setting:

```toml
PUBLIC_DEMO_MODE = "true"
```

In this mode, visitors can click the curated demo questions and see the RAG
workflow explanation without triggering provider embedding or chat calls. This is
the safest setting for a public portfolio link.

## Private Live RAG Secrets

For your own testing with real document upload, FAISS embedding, and LLM answer
generation, switch demo mode off and add a Google AI Studio Gemini key:

```toml
PUBLIC_DEMO_MODE = "false"
LLM_PROVIDER = "gemini"
GOOGLE_API_KEY = "your-google-ai-studio-key"
# GEMINI_API_KEY = "your-google-ai-studio-key" also works
# GOOGLE_AI_API_KEY = "your-google-ai-studio-key" also works
GEMINI_CHAT_MODEL = "gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
```

The key should be at the top level of Streamlit secrets. If the app still says
the key is missing, open `Secrets Diagnostics` in the sidebar. It shows only the
secret names visible to the app, never secret values.

OpenAI remains available for local comparison if you set
`LLM_PROVIDER = "openai"` and provide `OPENAI_API_KEY`, but the hosted portfolio
app should use Gemini or public demo mode to avoid OpenAI quota issues.

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
3. Ask one of the built-in demo questions:
   - `How should I solve an overdetermined system, and why is QR better than normal equations?`
   - `When should I use conjugate gradient instead of LU factorization?`
   - `What is the difference between Jacobi, Gauss-Seidel, and SOR?`
4. Show the sources under the answer.
5. Open `Agent Trace` to show query classification and retrieval rewriting.
6. Explain that the live RAG mode uses Gemini embeddings, FAISS, and LangGraph,
   while the public portfolio mode is locked to curated examples to prevent
   accidental API spending.
7. Explain that MCP export is separated from retrieval because export is an
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
