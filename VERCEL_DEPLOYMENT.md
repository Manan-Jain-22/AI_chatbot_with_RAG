# Vercel Deployment Notes

Vercel can host the lightweight web/API demo in this repo:

- `index.html` is the browser UI.
- `api/health.py` checks runtime status.
- `api/chat.py` calls the same LangGraph RAG pipeline used by Streamlit.

Vercel's Python runtime supports ASGI/WSGI apps and Python serverless functions in files under `api/`. It also supports `BaseHTTPRequestHandler` functions, which is what this project uses for the minimal API.

## Important Limitation

The Streamlit app is still the best interface for uploading course documents and rebuilding the FAISS index. Vercel functions are stateless serverless functions, so they are not a good place to accept user uploads and persist a FAISS index.

For a working Vercel demo, use one of these approaches:

1. Build a sanitized FAISS index locally and deploy it with the project.
2. Move vectors to a hosted database such as Pinecone, Weaviate, or pgvector.
3. Keep Vercel as a frontend/API shell and host the RAG backend elsewhere.

## Local Pre-Deployment Check

```bash
python -m compileall src app api
python -m src.vector_store
```

Then run the Streamlit app locally:

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
