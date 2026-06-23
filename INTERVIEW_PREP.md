# RAG Chatbot Interview Prep

## Resume Bullets

- Built a RAG chatbot using LangChain and FAISS for semantic retrieval from 100+ documents, enabling contextual responses.
- Designed LangGraph-based multi-step agent workflow with MCP tool access for query rewriting and response generation.

## 60-Second Project Pitch

I built a document-grounded chatbot that lets a user upload PDFs, text files, or Markdown files, indexes them with OpenAI embeddings in FAISS, and answers questions using retrieved context instead of relying only on the model's memory. LangChain handles the document loading, chunking, embeddings, vector store, and LLM calls. FAISS gives fast semantic similarity search over the document chunks. Streamlit provides the chat interface.

The agentic part is built with LangGraph. I model the RAG flow as explicit graph nodes: rewrite the user question, prepare the retrieval query through an MCP tool, retrieve relevant chunks from FAISS, generate a grounded draft answer, and format the final answer through another MCP tool. I also added a human approval/export step in the UI so the model does not automatically write outputs without review.

## Architecture

```text
User
  -> Streamlit UI
  -> LangGraph workflow
      -> OpenAI LLM rewrites the query
      -> MCP tool prepares retrieval query
      -> FAISS retrieves document chunks
      -> OpenAI LLM generates grounded answer
      -> MCP tool formats final answer
  -> User reviews answer
  -> Optional MCP-style export tool writes approved answer to Markdown
```

## Where Each Technology Is Used

**LangChain**

LangChain is the application framework around the RAG components. It provides:

- `PyPDFLoader` and `TextLoader` for document ingestion
- `RecursiveCharacterTextSplitter` for chunking
- `OpenAIEmbeddings` for creating vectors
- FAISS vector store integration
- `ChatOpenAI` for answer generation
- LangChain `Tool` abstractions used by the agent workflow

**FAISS**

FAISS is the vector database layer. Each document chunk is embedded into a numeric vector. At question time, the rewritten query is embedded and FAISS returns the most semantically similar chunks. This gives the LLM relevant context and reduces hallucination.

**OpenAI**

OpenAI is used in two places:

- embeddings: converts document chunks and queries into vectors
- chat model: rewrites questions and generates grounded answers

**LangGraph**

LangGraph is used because the workflow has multiple controlled steps rather than one single chain. The graph currently routes through:

```text
rewrite_query
prepare_query_with_mcp
retrieve
generate_answer
format_answer_with_mcp
```

This makes the system easier to debug and extend. For example, I can add approval, export, grading, retry, or query expansion nodes without rewriting the whole app.

**MCP**

MCP is used as the tool boundary. The project includes an MCP server in `src/mcp_server.py` that exposes tools for:

- preparing retrieval queries
- formatting final responses
- exporting approved answers

The graph can load these tools through `langchain-mcp-adapters` in `src/mcp_tools.py`. There are also same-schema local wrappers so the app can still run in restricted environments. The important design idea is that external capabilities are separated from LLM reasoning: the LLM decides or the graph routes, but tools perform deterministic actions.

**Streamlit**

Streamlit is the user-facing app. It supports:

- document upload
- FAISS index rebuild
- chat interface
- answer review
- approved answer export

## Why This Is Agentic

It is more than a single prompt call. The system maintains a controlled state and moves through a graph of actions:

1. understand and rewrite the user question
2. call a tool to prepare the search query
3. retrieve knowledge from FAISS
4. generate a draft answer
5. call a tool to format the answer
6. wait for human approval before export

That is the agentic layer: tool use, stateful multi-step control flow, and a human-in-the-loop action boundary.

## Strong Interview Answer For MCP

I used MCP to make tool access modular. Instead of baking every utility directly into the prompt or the LLM chain, I exposed deterministic capabilities as tools: retrieval-query preparation, response formatting, and approved answer export. LangGraph controls when those tools are called. This separation matters because it keeps LLM reasoning separate from side effects. For example, the model can draft an answer, but exporting is only triggered after user approval.

## Common Questions

**Why not just ask the LLM directly?**

Because the LLM may not know the user's private documents and may hallucinate. RAG grounds answers in retrieved chunks from the uploaded documents.

**Why chunk documents?**

Full documents are too large and too broad for retrieval. Chunking makes each vector represent a focused piece of information. Overlap helps preserve context across boundaries.

**Why FAISS?**

FAISS is fast, local, and simple for semantic similarity search. It is a strong fit for a project where I want local vector search without managing an external vector database service.

**Why LangGraph instead of a basic LangChain chain?**

LangGraph gives explicit control flow and state. That makes it easier to add tool calls, conditional routing, retries, review steps, and export steps.

**How do you prevent hallucinations?**

The generation prompt tells the model to answer only from retrieved context and cite sources. FAISS supplies context, and the final answer includes source labels.

**What would you improve next?**

- add answer quality grading and retry retrieval if context is weak
- add hybrid keyword plus vector search
- add persistent chat history
- add a real Google Docs MCP server for approved exports
- deploy the app with authentication
