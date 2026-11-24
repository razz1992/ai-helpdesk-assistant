# Architecture and responsibility boundaries

## Runtime flow
1. CLI receives an employee question.
2. LangGraph starts at `classify_query`.
3. A conditional edge selects one route:
   - `knowledge_question`: LangChain retriever, then prompt + chat model.
   - `ticket_status`: deterministic local ticket lookup.
   - `escalation_request`: simulated human escalation.
   - `out_of_scope`: safe fallback.
4. CLI prints the result.

## Framework responsibilities
| Framework | Responsibility |
|---|---|
| LangChain | Chat model, prompt, embeddings, document splitting, FAISS retrieval and tool abstraction |
| LangGraph | Shared state, nodes, edges, conditional routing and compiled graph |
| LangSmith | Tracing, trace inspection, datasets and evaluation experiments |

## Why use a deterministic classifier first?
The first graph lesson is State, Node, Edge and conditional routing. A deterministic classifier keeps branch selection easy to inspect. Replace it with a structured-output LLM classifier as a follow-up exercise.

## Human-in-the-loop extension
The current graph marks a conversation for simulated escalation. A later exercise can use a LangGraph interrupt and checkpointer to pause for approval and resume with the same thread ID. No real ticket is created.

## Limitations
- Initial classifier is rule-based.
- The in-memory vector store is rebuilt on startup.
- Escalation is simulated, not an actual approval UI.
- This is a learning scaffold, not a production helpdesk.
