# Project Progress — AI Helpdesk Assistant

**Last updated:** 2026-11-24  
**Project directory:** `E:\RazzFnFData\PostNGRWork\YQT1\AgenticLearning\MLOps\RAG\2\ai-helpdesk-assistant`  
**Purpose:** Learning/demo implementation of an AI IT helpdesk using LangChain, LangGraph, LangSmith and RAG.

## 1. Conversation actions and decisions

| # | Action / decision | Work completed | Status |
|---:|---|---|---|
| 1 | Defined the learning use case | Selected a synthetic IT helpdesk assistant to demonstrate LangChain components, LangGraph orchestration and LangSmith tracing/evaluation | Done |
| 2 | Scoped the demo behavior | Included knowledge-base Q&A, local mock-ticket lookup, simulated human escalation and out-of-scope fallback | Done |
| 3 | Created the project scaffold | Added application source, data, tests, evaluation starter and documentation directories | Done |
| 4 | Added synthetic knowledge data | Added five Markdown IT knowledge-base articles | Done |
| 5 | Added synthetic operational data | Added six ticket records, routing rules, twelve conversation scenarios, twelve evaluation examples and an escalation policy | Done |
| 6 | Implemented LangChain components | Added chat-model factory, prompts, retriever and ticket lookup tool | Done |
| 7 | Implemented LangGraph workflow | Added state, intent classification, retrieval/answer nodes, ticket lookup, escalation and fallback routing | Done; greeting intent still missing |
| 8 | Added project documentation | Added architecture, data contract and one-day learning plan documents | Done |
| 9 | Configured DeepSeek chat inference | Application configuration supports DeepSeek via its OpenAI-compatible API; secret is loaded from local environment | Implemented; live API call not yet verified |
| 10 | Investigated initial embedding failure | Runtime attempted to connect to Ollama, but Ollama was not installed/available on the Windows machine | Diagnosed |
| 11 | Selected a low-CPU embedding model | Chose `sentence-transformers/all-MiniLM-L6-v2` (small model, 384-dimensional embeddings) to avoid requiring an Ollama service | Done |
| 12 | Updated embedding configuration | Added `sentence-transformers` provider and model setting; kept Ollama/OpenAI as optional embedding providers | Done |
| 13 | Updated embedding implementation | Added `HuggingFaceEmbeddings` via `langchain-huggingface`; configured `device=cpu` and normalized embeddings | Done |
| 14 | Updated dependencies and environment template | Added `langchain-huggingface` and `sentence-transformers`; updated `.env.example` for CPU MiniLM defaults | Done |
| 15 | Updated local environment | Set local `.env` embedding provider/model while preserving existing credentials; secret values were not displayed or copied into this document | Done; local `.env` must remain untracked |
| 16 | Fixed pytest import discovery | Added `pytest.ini` with `pythonpath = .` and `testpaths = tests` after a collection error could not import `src` | Done |
| 17 | Validated Python source | `compileall` completed successfully for application source in the latest validation | Pass |
| 18 | Ran unit tests | `python -m pytest -q tests` completed with **3 passed**; 17 LangSmith dependency deprecation warnings remain | Pass with warnings |
| 19 | Smoke-tested MiniLM embeddings | Embedded a VPN question successfully; returned vector length was **384** | Pass |
| 20 | Checked project/Git state | Confirmed project directory was not yet a Git repository; GitHub CLI is installed and authenticated as the connected account | Done |
| 21 | Reviewed publish safety | `.gitignore` excludes `.env`, virtual environments, Python caches, local vector indexes and logs; `.env.example` is explicitly allowed | Reviewed |
| 22 | Corrected setup documentation | Updated README to describe DeepSeek chat + CPU MiniLM embeddings instead of the obsolete Ollama-only default | Done |
| 23 | Recorded progress | Created this project action/progress log | Done |

## 2. Current implementation

### Runtime flow

1. CLI receives a user question.
2. LangGraph calls the rule-based intent classifier.
3. The graph routes to knowledge retrieval, ticket lookup, simulated escalation or fallback.
4. Knowledge questions load Markdown articles, split them into chunks, embed them with MiniLM and build an in-memory FAISS retriever.
5. Retrieved context is passed to the configured chat model to generate a grounded answer.

### Current configuration

| Concern | Current choice |
|---|---|
| Chat provider | DeepSeek |
| Chat model | `deepseek-flash` |
| Embedding provider | Sentence Transformers |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding execution | CPU |
| Embedding dimensions | 384 |
| Vector store | FAISS, in-memory |
| Orchestration | LangGraph |
| Tracing/evaluation | LangSmith, optional |

## 3. Validation evidence

| Check | Result |
|---|---|
| Framework imports (LangChain, LangGraph, LangSmith, FAISS) | Passed in the recorded validation |
| Python compile check | Passed |
| Offline unit tests | 3 passed |
| MiniLM embedding smoke test | Passed; 384 dimensions |
| Live DeepSeek API call | Not yet verified |
| Full end-to-end RAG answer | Not yet verified |
| Greeting behavior (for example, `hi`) | Known issue: currently routed to `out_of_scope` |
| Git repository / first commit | Not yet created at the time this log was drafted |

The 17 warnings from the test run are deprecation warnings emitted by the installed LangSmith dependency around `asyncio.iscoroutinefunction`; they did not fail the tests.

## 4. Remaining work

### Priority 1 — complete functional verification
- [ ] Confirm the DeepSeek API key is configured locally without printing or exposing it.
- [ ] Run the CLI and verify a greeting, such as `hi`, receives a suitable greeting response.
- [ ] Add an explicit greeting intent and graph route, or otherwise handle greetings before the out-of-scope fallback.
- [ ] Run a real knowledge-base question (for example, VPN connection) through retrieval and DeepSeek answer generation.
- [ ] Verify ticket lookup and simulated escalation with representative inputs.
- [ ] Record actual end-to-end results and any fixes here.

### Priority 2 — improve quality
- [ ] Add unit tests for intent routing, retrieval and ticket/escalation behavior.
- [ ] Evaluate retrieval relevance and answer grounding against the supplied evaluation dataset.
- [ ] Review LangSmith tracing configuration and run an evaluation when credentials are available.
- [ ] Consider adding explicit confidence/empty-retrieval handling.

### Priority 3 — repository workflow
- [ ] Create a GitHub repository.
- [ ] Initialize Git locally, verify ignored files, and review staged files before committing.
- [ ] Create the initial commit and push the default branch.
- [ ] Verify the remote repository contains the expected source and documentation but no `.env`, API keys, `.venv`, caches or local FAISS artifacts.

## 5. Important constraints

- All ticket and helpdesk data in this project is synthetic.
- Escalation is only a simulated state change.
- The local `.env` contains configuration/secrets and must never be committed.
- FAISS is currently created in memory at runtime; there is no persisted index to migrate.
- The successful offline tests and embedding smoke test do not prove that live DeepSeek inference or the complete RAG path works.

## 6. Publish-safety review note

- A separate root-level `1.py` file was found outside the documented application structure. It contains a hard-coded credential-like value and was deliberately excluded from the Git index. It is still present locally and is not part of the initial project commit. Verify whether that value is a real API credential and rotate it if it is genuine. Do not publish that file or reuse the value.
