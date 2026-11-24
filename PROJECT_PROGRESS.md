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


## 7. Follow-up implementation and validation — 2026-11-24

| # | Action | Result | Status |
|---:|---|---|---|
| 24 | Added greeting intent | Added exact greeting matching for hi/hello/hey and common greetings, plus a dedicated greeting node and graph route | Done |
| 25 | Expanded deterministic routing tests | Added tests for greeting, VPN routing, sensitive credential requests, ticket lookup and escalation | Done |
| 26 | Added credential-disclosure guard | Requests for another employee's password/credentials now use a safe fallback instead of the password-help knowledge route | Done |
| 27 | Tested live DeepSeek connectivity | Request reached DeepSeek, which returned HTTP 402 Insufficient Balance; API generation cannot be verified until account balance is available | Blocked by provider billing |
| 28 | Tested real MiniLM + FAISS retrieval | VPN question returned three chunks; top two were from KB002 VPN troubleshooting | Pass |
| 29 | Tested graph end-to-end with a stub chat model | Actual classifier, MiniLM/FAISS retrieval and LangGraph answer-generation chain executed; only chat-model response was stubbed | Pass |
| 30 | Added local evaluation runner | Added per-case intent, route and retrieval@3 evaluation over the 12-case synthetic dataset | Done |
| 31 | Added LangSmith evaluation mode | Added optional dataset creation/reuse and an experiment for deterministic routing + retrieval; it does not invoke DeepSeek | Implemented; not run because LANGSMITH_API_KEY is not configured |
| 32 | Expanded automated tests | Latest run: **11 passed**, with 18 dependency deprecation warnings | Pass |
| 33 | Ran local evaluation | Intent accuracy **12/12 (100%)**; route accuracy **12/12 (100%)**; retrieval hit@3 **5/5 (100%)** for cases that actually use the knowledge route | Pass |
| 34 | Updated README | Documented greeting/security behavior, test commands, local evaluation and optional LangSmith evaluation | Done |

### Current verification boundaries

- The DeepSeek API endpoint was reached, but the response was HTTP 402 (Insufficient Balance). This proves the request reached the provider; it does not prove successful inference.
- The real MiniLM embedding model and FAISS retrieval path passed on the VPN question.
- The complete LangGraph knowledge path passed with a stubbed chat model. The actual DeepSeek-generated answer remains unverified.
- The LangSmith evaluation runner is implemented. The local `.env` currently has no configured `LANGSMITH_API_KEY`, so remote dataset upload and experiment execution remain pending.
- C09 is an escalation case and intentionally bypasses retrieval; it is excluded from the retrieval hit@3 denominator.
- Latest automated validation: `compileall` passed; `pytest -q tests` reported 11 passed.
- The current GitHub repository is public. Review the changed files and staged content before committing or pushing; do not stage the unrelated root-level `1.py` file.


## 8. Chat provider migration to Google Gemini — 2026-11-24

The project chat provider has been switched from DeepSeek to Google Gemini at the user's direction. The previous DeepSeek HTTP 402 result is retained above as historical troubleshooting evidence only; DeepSeek is no longer the configured provider.

| # | Action | Result | Status |
|---:|---|---|---|
| 35 | Inspected Google_ADK workspace configuration | Confirmed the Google ADK agent environment has a configured `GOOGLE_API_KEY` and `GOOGLE_GENAI_USE_VERTEXAI=0`; workspace README identifies `gemini-3.5-flash-lite` as the selected working model | Done |
| 36 | Reused Google API key | Copied the key locally from `Google_ADK\my_first_agent\.env` into this project's ignored `.env`; the key was not displayed or logged | Done |
| 37 | Changed chat provider defaults | Set `LLM_PROVIDER=google` and `GOOGLE_MODEL=gemini-3.5-flash-lite` in the project environment and configuration | Done |
| 38 | Added LangChain Google integration | Added `langchain-google-genai` to requirements and installed version 4.5.2 in the project virtual environment | Done |
| 39 | Implemented Gemini chat model factory | Added `ChatGoogleGenerativeAI` using `GOOGLE_API_KEY`; local MiniLM embedding implementation remains unchanged | Done |
| 40 | Removed old DeepSeek secret from project environment | Removed the `DEEPSEEK_API_KEY` line from the helpdesk project's local `.env`; no DeepSeek key is needed for the selected provider | Done |
| 41 | Tested Google API connectivity | Live invocation of `gemini-3.5-flash-lite` returned the expected `GEMINI_CONNECTION_OK` response | Pass |
| 42 | Tested full live VPN RAG path | Actual question passed through rule classifier, local MiniLM embeddings, FAISS top-3 retrieval and Google Gemini answer generation; received a grounded VPN procedure | Pass |
| 43 | Updated project documentation | Updated README and architecture documentation for Gemini chat + local MiniLM/FAISS retrieval | Done |

### Current active model configuration

- Chat provider: `google`
- Chat model: `gemini-3.5-flash-lite`
- Integration: `langchain-google-genai` (installed version 4.5.2)
- Credential variable: `GOOGLE_API_KEY` in local `.env` (ignored by Git)
- Embedding provider: `sentence-transformers`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: FAISS, in-memory

### Validation evidence

- Direct Gemini API smoke test: passed.
- Full VPN question → MiniLM → FAISS → Gemini answer: passed.
- The live answer included VPN connection checks, approved gateway selection, corporate sign-in/MFA, escalation criteria and support-ticket diagnostic details.
- The Google model reports that sampling temperature is fixed; the Gemini factory therefore omits the temperature parameter.
- LangSmith remains optional and separately configured. The local environment still needs `LANGSMITH_API_KEY` before remote evaluation can be run.


## 9. Fresh review and LangSmith Service Key tracing repair — 2026-11-24

| # | Action | Result | Status |
|---:|---|---|---|
| 44 | Revisited live project state | Confirmed Gemini + MiniLM + FAISS + LangGraph implementation, 12 automated tests and local evaluation | Done |
| 45 | Reproduced LangSmith 403 | Confirmed the Service Key can access organization metadata; the configured workspace UUID did not identify the visible Workspace 1 | Root cause identified |
| 46 | Retrieved organization/workspace identifiers safely | Used authenticated LangSmith organization/workspace APIs; no API key value was printed | Done |
| 47 | Verified Service Key role assignments | Workspace role resolves to WORKSPACE_ADMIN; organization role resolves to ORGANIZATION_USER | Sufficient for this tracing workflow |
| 48 | Verified scoped LangSmith SDK authentication | Client with the correct workspace ID and X-Organization-Id header authenticated successfully; project listing succeeded | Pass |
| 49 | Updated local .env | Corrected LANGSMITH_ORGANIZATION_ID and LANGSMITH_WORKSPACE_ID; API key unchanged; .env remains ignored by Git | Done |
| 50 | Updated LangSmith client configuration | Added organization/workspace settings and explicit Client construction; tracing_v2_enabled now uses the scoped client for graph and nested model runs | Done |
| 51 | Updated live trace verification script | Uses the same scoped client and retries transient trace-query failures while ingestion/indexing completes | Done |
| 52 | Updated README and architecture docs | Documented organization-scoped Service Key headers and live trace verification command | Done |
| 53 | Re-ran automated tests after code changes | 12 passed; only existing SDK/FAISS deprecation warnings remain | Pass |
| 54 | Re-ran local evaluation | Intent 12/12, route 12/12, retrieval hit@3 5/5 | Pass |
| 55 | Executed live Gemini RAG request | Gemini answered a VPN question using three retrieved MiniLM/FAISS documents | Pass |
| 56 | Inspected LangSmith trace records | Confirmed helpdesk_conversation root runs and nested ChatGoogleGenerativeAI LLM runs are queryable in ai-helpdesk-assistant | Pass |

### Current observability configuration

- LangSmith authentication uses an organization-scoped Service Key.
- The client sends X-Organization-Id and X-Tenant-Id (via the SDK workspace setting).
- The graph uses the explicitly scoped LangSmith client through tracing_v2_enabled.
- Expected trace hierarchy: helpdesk_conversation → classifier/router/retriever → prompt/chain → ChatGoogleGenerativeAI LLM.
- The local .env remains excluded from Git.

### Remaining verification

- Live verification command passed after retry/limit correction: Service Key authentication, one real Gemini request, root LangGraph run and nested Gemini LLM run were all confirmed queryable.
- In LangSmith, open Tracing Projects → ai-helpdesk-assistant and inspect a recent helpdesk_conversation trace.
- No Git commit or push was performed. The unrelated root-level 1.py remains untracked and must stay excluded.


## 10. Portfolio packaging and publication — 2026-11-24

| # | Action | Result | Status |
|---:|---|---|---|
| 57 | Re-ran Python compilation | `compileall` completed successfully for `src`, `tests`, `evaluation` and `scripts` | Pass |
| 58 | Re-ran automated tests | `pytest -q tests`: **12 passed**; 18 dependency deprecation warnings remain | Pass |
| 59 | Re-ran local evaluation | Intent **12/12 (100%)**; route **12/12 (100%)**; retrieval hit@3 **5/5 (100%)** | Pass |
| 60 | Re-ran live Gemini + LangSmith verification | Service Key authentication passed; real Gemini request completed; root `helpdesk_conversation` and nested LLM run were queryable in the configured LangSmith workspace | Pass |
| 61 | Added portfolio case study | Created `portfolio/README.md` with problem, architecture, stack, evidence, run instructions, scope limitations and security guidance | Done |
| 62 | Added portfolio evidence folder | Created `portfolio/screenshots/README.md` with a checklist for sanitized observability and validation screenshots | Done |
| 63 | Checked patch formatting | `git diff --check` passed; only normal Windows LF/CRLF notices appeared | Pass |
| 64 | Prepared explicit Git staging list | Excluded `.env` and unrelated root-level `1.py`; only intended application, documentation, scripts and portfolio files are eligible | Done |
| 65 | Committed and pushed project changes | Commit `240adf2` (`Add Gemini RAG and LangSmith observability`) pushed to `origin/main`; remote HEAD verified against the commit | Pass |

### Completion state

- Gemini + local MiniLM/FAISS + LangGraph application is runnable locally.
- Live LangSmith tracing is verified for a real Gemini-backed knowledge question.
- Automated tests and local evaluation passed on 2026-11-24.
- Portfolio case-study folder is created; sanitized screenshots can be added to `portfolio/screenshots/`.
- Git commit `240adf2` (`Add Gemini RAG and LangSmith observability`) was pushed successfully to `origin/main` on 2026-11-24.
- Final working tree contains only the unrelated root-level `1.py` as untracked; it remains excluded from the commit and push.


## 11. Persistent conversation memory with LangGraph SQLite checkpointer — 2026-11-24

| # | Action | Result | Status |
|---:|---|---|---|
| 66 | Reviewed current graph and state | Confirmed one graph invocation per question, no checkpointer and no message history | Done |
| 67 | Installed SQLite checkpoint integration | Installed `langgraph-checkpoint-sqlite 3.1.3`; added bounded dependency to requirements | Done |
| 68 | Added checkpoint-capable graph factory | `build_graph(checkpointer=None)` now compiles with an optional saver | Done |
| 69 | Added message history to shared state | Added `messages` with LangGraph `add_messages` reducer; user and assistant turns are appended | Done |
| 70 | Added context-aware follow-up baseline | Common follow-up phrases are linked to the previous user question before knowledge retrieval | Done |
| 71 | Updated knowledge prompt | Prior conversation messages are passed to the knowledge-answer chain | Done |
| 72 | Updated CLI | New or existing conversation IDs are accepted; SQLite checkpoints persist in `data/conversations.sqlite` | Done |
| 73 | Protected local checkpoint files | Added SQLite/DB file patterns to `.gitignore` | Done |
| 74 | Added persistence tests | Added database close/reopen test and thread-isolation test | Done |
| 75 | Ran compile and offline tests | Compilation passed; `pytest -q tests`: **14 passed**, 18 existing dependency deprecation warnings | Pass |
| 76 | Added live observability integration test | Added two-turn Gemini + SQLite + LangSmith trace verification script | Done; execution pending |

### Module 1 implementation notes

- Checkpointing is keyed by `thread_id`; using a new ID isolates a new conversation.
- The SQLite database stores graph checkpoints and message state; it is local runtime data and must not be committed.
- LangSmith tracing continues to use the existing organization/workspace-scoped Service Key client. Each turn is a separate root run with conversation thread metadata.
- The follow-up resolver is intentionally rule-based and narrow. It demonstrates contextual follow-up handling without claiming to be a general memory or coreference solution.
- No API key, Service Key or `.env` value was added to source or documentation.
- The unrelated root-level `1.py` remains untracked and excluded from all Git operations.


| 77 | Reviewed live CLI restart experiment and LangSmith dashboard | Same conversation ID resumed after restart; dashboard showed 35 successful traces and zero failed traces | Confirmed by user |
| 78 | Fixed natural-language next-step follow-up | Added "what should I try next?" variants and backward search for the latest substantive IT topic across earlier user turns | Done |
| 79 | Added multi-turn regression coverage | Test covers VPN question → tried steps → next-step question | Done |
| 80 | Re-ran test suite and whitespace validation | **16 passed**, 18 dependency warnings; `git diff --check` exit code 0 | Pass |
