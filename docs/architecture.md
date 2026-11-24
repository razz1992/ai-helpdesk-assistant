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


## Persistent conversation memory — Module 1

The graph can optionally compile with a LangGraph checkpointer. The CLI uses
`SqliteSaver` and stores checkpoints in `data/conversations.sqlite`. That file is
local runtime data and is excluded from Git.

- `thread_id` identifies one conversation. Reusing it resumes the saved graph state.
- A different `thread_id` starts an isolated conversation.
- `HelpdeskState.messages` uses LangGraph's `add_messages` reducer to append
  HumanMessage and AIMessage turns instead of replacing the list.
- The graph captures the user turn before classification and appends the assistant
  answer before ending the turn.
- The knowledge-answer prompt receives earlier messages as conversation history.
- A small deterministic follow-up resolver enriches common phrases such as
  "I already tried the first two steps" with the previous user question before
  retrieval. This is a learning baseline, not a general coreference model.
- LangSmith receives the thread ID in run configuration and records each turn as
  a separate `helpdesk_conversation` root run.

The checkpointer persists graph state; it does not itself summarize history,
perform semantic memory retrieval, or guarantee that every node uses history.
Those are separate application design decisions.

### Local validation

```powershell
python -m pytest -q tests
python -m scripts.test_persistent_conversation
```

The persistence test closes and reopens the SQLite connection between turns,
then verifies that the same thread resumes with four messages (two user and
two assistant messages). The live integration script additionally makes two
Gemini-backed turns and checks for both root runs and nested LLM runs in LangSmith.


### Follow-up resolver refinement

The resolver recognises natural next-step questions such as "What should I try next?"
and searches backward through earlier user turns for the latest substantive IT topic.
This matters when the conversation already contains an intermediate follow-up such as
"I already tried the first two steps." It avoids treating that short follow-up as the
original topic. Regression tests cover this multi-turn sequence.
