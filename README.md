# AI Helpdesk Assistant — LangChain + LangGraph + LangSmith

A synthetic IT helpdesk assistant built as a hands-on learning project for LangChain, LangGraph, LangSmith, retrieval-augmented generation (RAG), and evaluation.

## What the demo does

- Answers supported IT questions using a small Markdown knowledge base and FAISS retrieval.
- Looks up synthetic incident records from local JSON data.
- Marks unresolved issues for simulated human escalation.
- Uses a deterministic fallback for unsupported requests.
- Provides starter evaluation data and documentation.

All tickets and knowledge-base records are fictional. No real service desk or ticketing system is contacted.

## Architecture

- **Chat model:** DeepSeek through its OpenAI-compatible API.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`, executed locally on CPU.
- **Vector store:** FAISS, built in memory when the first knowledge question is processed.
- **Orchestration:** LangGraph state graph.
- **LLM framework:** LangChain.
- **Tracing/evaluation:** LangSmith integration (optional).

The chat and embedding providers are independent. Ollama is not required for the default embedding configuration.

## Prerequisites

- Windows 10/11
- Python 3.14 (the current project environment uses Python 3.14)
- Git
- A DeepSeek API key for live answer generation

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

Copy the safe template and add your own credentials:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set the DeepSeek key:

```dotenv
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_actual_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
```

The default local embedding settings are:

```dotenv
EMBEDDING_PROVIDER=sentence-transformers
SENTENCE_TRANSFORMER_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

The MiniLM model is downloaded from Hugging Face on first use and then loaded locally. CPU execution is explicitly configured in the embedding factory. An internet connection is required for the initial model download.

LangSmith tracing is optional. To enable it, set `LANGSMITH_TRACING=true` and provide `LANGSMITH_API_KEY`. Do not commit the real `.env` file or paste secrets into source code.

## Run

```powershell
python -m src.main
```

The first knowledge-base query splits the Markdown articles, embeds the chunks, and creates an in-memory FAISS index. The index is rebuilt on each new application process.

## Validate

```powershell
python -m compileall -q src
python -m pytest -q
```

The current offline test suite contains three tests. Live DeepSeek generation should be validated separately with a configured API key.

## Project layout

- `data/knowledge_base/`: five synthetic IT knowledge articles.
- `data/tickets.json`: six mock ticket records.
- `data/routing_rules.json`: intent and routing reference data.
- `data/conversation_scenarios.json`: twelve manual scenarios.
- `data/evaluation_dataset.json`: twelve evaluation examples.
- `data/escalation_policy.md`: simulated escalation policy.
- `src/langchain_components/`: model/embedding factories, prompts, retriever and ticket tool.
- `src/langgraph_workflow/`: graph state, nodes and conditional routing.
- `evaluation/`: evaluation dataset loader and evaluation starter.
- `docs/`: architecture, data contract and learning plan.
- `PROJECT_PROGRESS.md`: action log, validation evidence and remaining work.

## Security and demo boundaries

- `.env` is excluded by `.gitignore`; `.env.example` is safe to publish.
- The virtual environment, caches, local vector indexes and logs are excluded from Git.
- Never commit API keys, tokens, passwords or other credentials.
- Ticket lookup uses only local synthetic JSON data.
- Escalation is simulated; no real ticket or external notification is created.
- The assistant must not invent ticket status or unsupported knowledge-base facts.
