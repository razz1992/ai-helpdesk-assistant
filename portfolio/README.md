# AI Helpdesk Assistant — Portfolio Case Study

## Project overview

A synthetic IT helpdesk assistant demonstrating a retrieval-augmented generation (RAG) workflow with deterministic intent routing, LangGraph orchestration, Google Gemini response generation, local semantic retrieval, and LangSmith observability.

**Project status:** Working end-to-end demo; live Gemini-to-LangSmith tracing verified on 24 November 2026.

## Problem and objective

IT support teams repeatedly answer knowledge-base questions, check ticket status, and triage escalation requests. This project demonstrates how an AI assistant can route these request types, retrieve relevant support guidance, and generate grounded answers while preserving trace-level visibility into execution.

## Architecture

```text
User question
    |
    v
LangGraph workflow
    |
    v
Deterministic intent classification
    |-- greeting ------------> greeting response
    |-- knowledge question --> MiniLM embeddings -> FAISS top-k -> Gemini answer
    |-- ticket status -------> synthetic ticket lookup
    |-- escalation ----------> simulated escalation response
    `-- out of scope --------> safe fallback

LangSmith tracing observes the graph execution and nested LangChain / Gemini calls.
```

## Technology stack

| Layer | Technology | Responsibility |
|---|---|---|
| Chat generation | Google Gemini `gemini-3.5-flash-lite` | Generates natural-language helpdesk answers |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Encodes local knowledge-base chunks (384 dimensions) |
| Vector search | FAISS | Retrieves semantically relevant chunks |
| Orchestration | LangGraph | Models state, nodes, and conditional routes |
| LLM framework | LangChain | Composes prompts, retrievers, and model calls |
| Observability | LangSmith | Captures graph and nested LLM traces |
| Evaluation | Local evaluation runner + pytest | Measures routing and retrieval behavior |

## Validation results

Latest validation run: 24 November 2026.

| Check | Result |
|---|---:|
| Python compilation | Passed |
| Automated tests | 12 / 12 passed |
| Intent accuracy | 12 / 12 (100%) |
| Route accuracy | 12 / 12 (100%) |
| Retrieval hit@3 | 5 / 5 (100%) |
| Live Gemini RAG request | Passed |
| LangSmith Service Key authentication | Passed |
| LangSmith root graph trace | Found |
| Nested Gemini LLM run | Found (1) |

The retrieval metric is calculated only for the five evaluation cases that use the knowledge route. Ticket, greeting, escalation, and fallback routes are not retrieval cases.

## LangSmith observability

LangSmith project: `ai-helpdesk-assistant`.

The live verification script authenticates with the configured organization-scoped Service Key, invokes a real Gemini-backed knowledge question, and queries LangSmith for:

- The root `helpdesk_conversation` LangGraph run.
- At least one nested LLM run for the Gemini model invocation.

Open [LangSmith](https://smith.langchain.com), select the configured workspace, then open **Tracing Projects → ai-helpdesk-assistant**. Expand a recent `helpdesk_conversation` run to inspect its child operations, timings, inputs/outputs, and model call.

## Run locally

From the repository root in PowerShell:

```powershell
python -m src.main
```

Example knowledge-base question:

```text
How do I connect to the company VPN?
```

Run automated tests:

```powershell
python -m pytest -q tests
```

Run local evaluation:

```powershell
python -m evaluation.evaluate
```

Verify live Gemini + LangSmith tracing:

```powershell
python -m scripts.test_langsmith_tracing
```

## Portfolio evidence to capture

Add screenshots to `portfolio/screenshots/` (do not include credentials or sensitive trace payloads):

1. LangSmith project run list showing recent `helpdesk_conversation` runs.
2. Expanded trace tree showing the root graph and nested Gemini LLM call.
3. Terminal output showing the automated test and local evaluation results.
4. Terminal output showing the live LangSmith verification script passing.

## Scope and limitations

- Ticket records and knowledge articles are synthetic demonstration data.
- Escalation is simulated; the application does not create a real ITSM ticket.
- FAISS is built locally/in memory; this is not a production persistence or distributed vector database design.
- Evaluation is a small deterministic test set, not a statistically representative production benchmark.
- The current application is a local demo; authentication, authorization, deployment, operational controls, and production data governance would be required before real enterprise use.

## Security notes

- Keep `.env` local and excluded from Git.
- Never commit API keys, Service Keys, workspace identifiers that are treated as internal configuration, or raw trace exports containing sensitive data.
- Do not add the unrelated root-level `1.py` file to this repository commit.

## Repository navigation

- `src/` — application, LangChain components, and LangGraph workflow.
- `data/knowledge_base/` — synthetic helpdesk knowledge articles.
- `data/tickets.json` — synthetic ticket records.
- `evaluation/` — local evaluation runner and cases.
- `scripts/` — live LangSmith tracing verification.
- `tests/` — automated tests.
- `docs/architecture.md` — architecture notes.
- `PROJECT_PROGRESS.md` — implementation history and completion log.
