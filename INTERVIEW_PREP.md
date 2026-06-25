# Computational Linear Algebra RAG Interview Prep

## Resume Bullets

- Built a RAG chatbot using LangChain and FAISS for semantic retrieval from 100+ documents, enabling contextual responses.
- Designed LangGraph-based multi-step agent workflow with MCP tool access for query rewriting & response generation.

## 60-Second Pitch

I built a Computational Linear Algebra study assistant rather than a generic chatbot over PDFs. The motivation was that in computational math, students often do not just need to remember a formula; they need to identify which numerical method applies to a problem and why.

The corpus comes from Computational Linear Algebra course material: lecture notes, formula sheets, solved examples, implementation notes, assignment review notes, and my own summaries. It covers direct methods for linear systems, sparse and banded systems, Jacobi, Gauss-Seidel, SOR, conjugate gradient, QR factorization, SVD, least squares, and eigenvalue methods.

The system uses LangChain for loading and chunking documents, OpenAI embeddings for semantic representations, FAISS for vector search, LangGraph for the multi-step workflow, Streamlit for the UI, and MCP for controlled export after a useful answer is finalized.

## Architecture

```text
Course documents
  -> LangChain loaders
  -> topic and section metadata
  -> section-aware chunking
  -> OpenAI embeddings
  -> FAISS vector index

Student question
  -> LangGraph query classification
  -> LangGraph query rewriting
  -> FAISS retrieval
  -> context selection
  -> grounded answer generation
  -> optional MCP export to Notion study card
```

## Why This Is Not Just Keyword Search

A student might ask: "How do I solve an overdetermined system?"

The relevant material could appear under:

- least squares
- QR factorization
- normal equations
- orthogonal projection
- SVD

Keyword search can miss that connection. Embeddings help because they retrieve by meaning, not just exact words.

## Document Processing

Each file becomes a LangChain `Document` with metadata:

- source file
- page number when available
- file type
- inferred course topic
- section heading

The project uses section-aware chunking. This matters for math notes because formulas without headings and surrounding explanation are hard to retrieve and hard for the LLM to use correctly.

## FAISS And Retrieval

Each chunk is embedded with OpenAI embeddings and stored in FAISS. At query time, the question is embedded with the same model, and FAISS returns the top-k semantically similar chunks.

FAISS is a good fit because it is fast, local, and simple for a course-document assistant. If this became a multi-user production system, I would consider pgvector, Pinecone, Weaviate, or another hosted vector database.

## LangGraph Workflow

The graph is split into explicit steps:

```text
classify_query
rewrite_query
retrieve
select_context
generate_answer
```

The query classification node identifies the likely topic, such as least squares or iterative methods. The query rewriting node expands vague student wording into retrieval-friendly terms. For example, "How do I solve an overdetermined system?" can be rewritten with terms like least squares, QR factorization, normal equations, SVD, and overdetermined linear systems.

After retrieval, the context selection node prioritizes chunks matching the classified topic. The answer generation node is instructed to answer only from retrieved course context and cite sources.

## MCP Usage

I kept MCP separate from the core retrieval logic. FAISS handles retrieval. LangGraph handles workflow control. MCP handles controlled external export after the answer is finalized.

The MCP server exposes:

- `export_answer_to_markdown`
- `save_study_card_to_notion`

The Notion tool saves a useful answer as a structured study card with:

- question
- answer
- topic
- tags
- source references

This separation is important because exporting to Notion is a side effect. It should happen only after the user reviews and confirms the answer.

## Retrieval Evaluation

I added a small golden evaluation set of representative student-style questions. Each row includes:

- question
- expected topic
- expected section
- expected source
- key answer points

The evaluation script measures precision@k and recall@k for retrieval. Recall matters because if the right course section is missing from the retrieved context, the LLM cannot answer reliably. Precision matters because too many loosely related chunks can confuse the model, especially for similar methods like Jacobi, Gauss-Seidel, SOR, and conjugate gradient.

## Strong Interview Answer

The biggest thing I learned is that useful RAG is not just connecting an LLM to a vector database. The quality depends heavily on preprocessing, metadata, chunking, retrieval evaluation, and deciding when the model should answer from text versus when it should call a tool. In this project, I used FAISS for semantic retrieval, LangGraph for structured workflow, and MCP only for controlled external export after user confirmation.

## Likely Follow-Ups

**Why section-aware chunking?**

Mathematical formulas need nearby explanations, assumptions, and headings. Section-aware chunking improves retrieval because a chunk carries more conceptual context.

**Why query rewriting?**

Students ask vague questions. Rewriting expands those questions into terms that appear in the course notes, improving recall.

**Why classify the query before retrieval?**

Topic classification helps context selection. If the question is about least squares, retrieved chunks from QR/SVD may be useful, but unrelated eigenvalue chunks should be deprioritized.

**Why MCP for Notion export?**

Notion is an external side-effecting system. MCP gives a clean tool boundary so the graph can keep retrieval and reasoning separate from external writes.

**What would you improve next?**

I would add deterministic Python/NumPy tool use for numerical verification. For example, if a user provides a matrix and asks whether Cholesky is appropriate, the assistant could retrieve the concept from the notes and then call a Python tool to check symmetry and positive definiteness.
