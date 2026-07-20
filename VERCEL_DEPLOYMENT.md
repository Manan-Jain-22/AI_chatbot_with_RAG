# Vercel Deployment Notes

The complete project application is the Streamlit app. It is the app to demo
because it supports document upload, FAISS index rebuilding, chat, answer
review, and MCP export in one place.

Vercel can host a lightweight web/API surface in this repo:

- `public/index.html` is the browser UI.
- `api/health.py` checks runtime status.
- `api/chat.py` calls the same LangGraph RAG pipeline used by Streamlit.

Vercel's Python runtime supports ASGI/WSGI apps and Python serverless functions in files under `api/`. It also supports `BaseHTTPRequestHandler` functions, which is what this project uses for the minimal API.

## Vercel Project Settings

Use these settings in the Vercel dashboard:

```text
Framework Preset: Other
Root Directory: ./
Build Command: leave empty
Output Directory: public
Install Command: pip install -r requirements-vercel.txt
```

If Vercel auto-fills an output directory like `.vercel/output` or a build command for a frontend framework, clear it. This repo is a static HTML page plus Python serverless functions, not a Next.js project.

## Why Vercel Is Not The Main App Yet

Vercel serverless functions are stateless. They are not a good place to accept
course document uploads, persist those files, and rebuild a FAISS index that
should survive across user sessions.

That is why this project currently has one complete app, Streamlit, and one
optional Vercel deployment surface for later hosting work.

For a production-style Vercel version, use one of these approaches:

1. Build a sanitized FAISS index locally and deploy it with the project.
2. Move vectors to a hosted database such as Pinecone, Weaviate, or pgvector.
3. Keep Vercel as a frontend/API shell and host the RAG backend elsewhere.

The cleanest future direction is option 2: keep the Vercel UI and Python API,
but replace local FAISS persistence with a hosted vector database.

## Local Pre-Deployment Check

```bash
python -m compileall src app api
python -m src.vector_store
```

Then run the complete Streamlit application:

```bash
streamlit run app/streamlit_app.py
```

## Vercel Environment Variables

Set these in the Vercel project dashboard:

```text
OPENAI_API_KEY
CHAT_MODEL
EMBEDDING_MODEL
FAISS_INDEX_DIR
```

Optional Notion export variables:

```text
NOTION_API_KEY
NOTION_DATABASE_ID
NOTION_API_VERSION
```

## Deploy Later

After GitHub auth is fixed and the latest commit is pushed:

```bash
vercel
vercel --prod
```
