# AI Helpdesk Assistant — LangChain + LangGraph + LangSmith

A synthetic IT helpdesk assistant built as a hands-on learning project for LangChain, LangGraph, LangSmith, retrieval-augmented generation (RAG), and evaluation.

## What the demo does

- Answers supported IT questions using a small Markdown knowledge base and FAISS retrieval.
- Uses Google Gemini for chat answer generation.
- Looks up synthetic incident records from local JSON data.
- Marks unresolved issues for simulated human escalation.
- Handles greetings separately from unsupported requests.
- Rejects requests for another person's credentials with a safe fallback.
- Provides local and optional LangSmith workflow evaluation.

All tickets and knowledge-base records are fictional. No real service desk or ticketing system is contacted.

## Architecture

- **Chat model:** Google Gemini `gemini-3.5-flash-lite` via `langchain-google-genai`.
- **Chat API key:** Reused from the local Google ADK project configuration; stored only in this project's ignored `.env`.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`, executed locally on CPU.
- **Vector store:** FAISS, built in memory when the first knowledge question is processed.
- **Orchestration:** LangGraph state graph.
- **LLM framework:** LangChain.
- **Tracing/evaluation:** LangSmith integration (optional).

The chat and embedding providers are independent. Gemini is used for answer generation; embeddings remain local and do not call Google's embedding API. Ollama is not required for the default configuration.

## Prerequisites

- Windows 10/11
- Python 3.14 (the current project environment uses Python 3.14)
- Git
- A Google AI Studio API key with access to the configured Gemini model

## Setup (PowerShell)

Run from the project directory:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the existing `.venv` is already present, activate it instead of recreating it.

## Configuration

The project uses the Google AI Studio API key configured in the separate `Google_ADK\my_first_agent\.env` workspace. For a new machine, copy `.env.example` to `.env` and set:

```dotenv
LLM_PROVIDER=google
GOOGLE_API_KEY=your_google_ai_studio_api_key
GOOGLE_MODEL=gemini-3.5-flash-lite
```

The default local embedding settings are:

```dotenv
EMBEDDING_PROVIDER=sentence-transformers
SENTENCE_TRANSFORMER_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

The MiniLM model is downloaded from Hugging Face on first use and then loaded locally. CPU execution is explicitly configured in the embedding factory. An internet connection is required for the initial model download.

LangSmith tracing/evaluation is optional. For the organization-scoped Service Key (`lsv2_sk_...`) used by this project, configure `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_ORGANIZATION_ID`, `LANGSMITH_WORKSPACE_ID`, and `LANGSMITH_PROJECT` in `.env`. The application creates an explicitly scoped LangSmith client and supplies both organization and workspace headers so the root LangGraph run and nested Gemini LLM calls are traced to the intended workspace. Do not commit the real `.env` file or paste secrets into source code.

## Run

```powershell
python -m src.main
```

The first knowledge-base query splits the Markdown articles, embeds the chunks, and creates an in-memory FAISS index. The index is rebuilt on each new application process.

## Validate and evaluate

```powershell
python -m compileall -q src tests evaluation
python -m pytest -q tests
python -m evaluation.evaluate

# One live Gemini request + LangSmith root/nested trace verification
python -m scripts.test_langsmith_tracing
```

The local evaluation runs the 12 synthetic examples and reports intent accuracy, graph route accuracy, and retrieval hit@3 for cases routed through knowledge retrieval. It does not call Gemini. Escalation cases intentionally bypass retrieval.

To upload the dataset and run the deterministic routing/retrieval experiment in LangSmith:

```powershell
python -m evaluation.evaluate --langsmith
```

This LangSmith experiment evaluates intent, route and MiniLM/FAISS retrieval; it does not invoke the chat model. Live Gemini answer generation is validated separately.

## Project layout

- `data/knowledge_base/`: five synthetic IT knowledge articles.
- `data/tickets.json`: six mock ticket records.
- `data/routing_rules.json`: intent and routing reference data.
- `data/conversation_scenarios.json`: twelve manual scenarios.
- `data/evaluation_dataset.json`: twelve evaluation examples.
- `data/escalation_policy.md`: simulated escalation policy.
- `src/langchain_components/`: model/embedding factories, prompts, retriever and ticket tool.
- `src/langgraph_workflow/`: graph state, nodes and conditional routing.
- `evaluation/`: evaluation dataset loader and local/LangSmith evaluation runner.
- `docs/`: architecture, data contract and learning plan.
- `portfolio/`: project case study and a checklist for sanitized portfolio screenshots.
- `PROJECT_PROGRESS.md`: action log, validation evidence and remaining work.

## Security and demo boundaries

- `.env` is excluded by `.gitignore`; `.env.example` is safe to publish.
- The virtual environment, caches, local vector indexes and logs are excluded from Git.
- Never commit API keys, tokens, passwords or other credentials.
- Ticket lookup uses only local synthetic JSON data.
- Escalation is simulated; no real ticket or external notification is created.
- The assistant must not invent ticket status or unsupported knowledge-base facts.


## Persistent conversations (Module 1)

The CLI now supports SQLite-backed LangGraph checkpoints and message history.

```powershell
python -m src.main
```

At startup, leave the conversation ID blank to create a new conversation, or
paste a previously displayed ID to resume it. Keep the ID if you want to continue
after closing and restarting the application. Checkpoints are stored locally in
`data/conversations.sqlite`; the database is ignored by Git.

The graph stores user and assistant messages using the `add_messages` reducer.
The knowledge-answer prompt receives previous turns, and a small rule-based
follow-up resolver links common follow-up phrases to the previous IT question.

Validate the persistence behavior offline:

```powershell
python -m pytest -q tests
```

Run the live two-turn Gemini + SQLite + LangSmith integration check (uses the
configured Gemini API and sends traces to the configured LangSmith project):

```powershell
python -m scripts.test_persistent_conversation
```

This is conversation checkpointing, not long-term semantic memory. The follow-up
resolver is intentionally narrow and will be improved in later learning modules.


## Human-in-the-loop approval (Module 2)

Escalation requests now pause at a LangGraph `interrupt()` node and wait for a
human decision in the CLI. The decision resumes the same SQLite-backed
conversation using `Command(resume=...)` and the original conversation ID.

- Approve (`y`): continue to the simulated escalation response.
- Reject (`n`): return a rejection response without taking an escalation action.

Run `python -m src.main` and enter:
`Please escalate this unresolved issue to a human.`

Try both approval and rejection in separate conversations. This is a learning
simulation only; it does not create an actual helpdesk ticket or notify a team.

Run the regression tests with `python -m pytest -q tests`.
