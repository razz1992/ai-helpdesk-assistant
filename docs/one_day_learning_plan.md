# One-day learning plan

## Session 1 — LangChain (60 minutes)
- Inspect model factory, prompt template, document loader, splitter and retriever.
- Understand embeddings and top-k retrieval.
- Inspect the ticket lookup function and LangChain tool wrapper.

## Session 2 — LangGraph core (90 minutes)
- Read HelpdeskState.
- Trace classify_query as a node.
- Inspect conditional edge and route_by_intent.
- Trace knowledge, ticket, escalation and fallback branches.

## Session 3 — Stateful workflows (60 minutes)
- Add a checkpointer and thread ID.
- Learn how state is saved and resumed.
- Add a simulated human approval interrupt after escalation.

## Session 4 — LangSmith (90 minutes)
- Configure LANGSMITH_TRACING and LANGSMITH_API_KEY.
- Run the app and inspect traces, prompts, retrieved documents and latency.
- Create a dataset from evaluation_dataset.json.
- Run an experiment with route correctness and answer quality evaluators.

## Session 5 — Testing and review (30 minutes)
- Run the 12 scenarios.
- Confirm ticket and routing behaviour.
- Remove a relevant KB article and inspect the trace.
- Record what each framework contributed.
