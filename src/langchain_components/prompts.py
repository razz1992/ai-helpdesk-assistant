"""Prompt templates used by individual LangChain capabilities."""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

KNOWLEDGE_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful internal IT helpdesk assistant. Answer only using "
            "the supplied knowledge-base context. Use prior conversation messages "
            "to understand references such as 'those steps' or 'it'. If the answer "
            "is not present, say the knowledge base does not contain it and recommend "
            "the approved support route. Never ask for or reveal passwords, MFA codes, "
            "recovery codes or private keys. Do not invent policy.",
        ),
        MessagesPlaceholder(variable_name="history", optional=True),
        (
            "human",
            "Question: {question}\n\n"
            "Knowledge-base context:\n{context}\n\n"
            "Give a concise, actionable answer and mention any relevant escalation.",
        ),
    ]
)
