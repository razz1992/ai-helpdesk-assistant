# Architecture and responsibility boundaries

## Runtime flow
1. CLI receives an employee question.
2. LangGraph starts at `classify_query`.
3. A conditional edge selects one route:
   - `greeting`: deterministic greeting response.
   - `knowledge_question`: MiniLM embeddings, FAISS retrieval, then a grounded Gemini answer.
   - `ticket_status`: deterministic local ticket lookup.
   - `escalation_request`: simulated human escalation.
   - `out_of_scope`: safe fallback.
4. CLI prints the result.

## Model and retrieval configuration

| Component | Implementation | Execution |
|---|---|---|
| Chat generation | Google Gemini `gemini-3.5-flash-lite` via `langchain-google-genai` | Google AI Studio API |
| Chat credentials | `GOOGLE_API_KEY` loaded from project `.env` | Local secret file, ignored by Git |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Local CPU |
| Vector store | FAISS | In-memory |
| Retrieval | Top 3 chunks | Local |

The Google chat model and local embedding model are independent. A VPN question is embedded locally, matched against the local KB, and the retrieved context is sent to Gemini for answer generation.

## Framework responsibilities

| Framework | Responsibility |
|---|---|
| LangChain | Chat model, prompt, embeddings, document splitting, FAISS retrieval and tool abstraction |
| LangGraph | Shared state, nodes, edges, conditional routing and compiled graph |
| LangSmith | Tracing, trace inspection, datasets and evaluation experiments |

For organization-scoped LangSmith Service Keys, the runtime constructs a `langsmith.Client` with the organization ID in `X-Organization-Id` and the workspace ID in `X-Tenant-Id`. `tracing_v2_enabled` uses that client for the complete LangGraph execution, including nested LangChain prompt and Gemini model runs. This avoids relying on the default environment-created client, which does not carry the required organization header.

## Why use a deterministic classifier first?
The first graph lesson is State, Node, Edge and conditional routing. A deterministic classifier keeps branch selection easy to inspect. Replace it with a structured-output LLM classifier as a follow-up exercise.

## Human-in-the-loop extension
The current graph marks a conversation for simulated escalation. A later exercise can use a LangGraph interrupt and checkpointer to pause for approval and resume with the same thread ID. No real ticket is created.

## Limitations
- Initial classifier is rule-based.
- The in-memory vector store is rebuilt on startup.
- Escalation is simulated, not an actual approval UI.
- This is a learning scaffold, not a production helpdesk.
