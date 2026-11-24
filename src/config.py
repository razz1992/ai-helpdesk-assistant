"""Central configuration loaded from environment variables."""
from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
TICKETS_FILE = DATA_DIR / "tickets.json"

load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    # Chat model configuration
    provider: str = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv(
        "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
    )
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Embeddings are independent of the chat provider.
    embedding_provider: str = os.getenv(
        "EMBEDDING_PROVIDER", "sentence-transformers"
    ).strip().lower()
    sentence_transformer_model: str = os.getenv(
        "SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    openai_embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL", "http://localhost:11434"
    )
    ollama_embedding_model: str = os.getenv(
        "OLLAMA_EMBEDDING_MODEL", "all-minilm"
    )

    temperature: float = float(os.getenv("TEMPERATURE", "0"))
    langsmith_project: str = os.getenv(
        "LANGSMITH_PROJECT", "ai-helpdesk-assistant"
    )


settings = Settings()


def validate_provider_config() -> None:
    """Validate chat and embedding provider configuration."""
    if settings.provider == "deepseek" and (
        not settings.deepseek_api_key
        or settings.deepseek_api_key == "PASTE_YOUR_DEEPSEEK_API_KEY_HERE"
    ):
        raise RuntimeError(
            "DEEPSEEK_API_KEY is missing. Add your key to the project .env file."
        )
    if settings.provider == "openai" and not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from the project .env file.")
    if settings.provider not in {"deepseek", "openai", "ollama"}:
        raise ValueError(
            "LLM_PROVIDER must be 'deepseek', 'openai' or 'ollama'."
        )

    if settings.embedding_provider == "openai" and not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai."
        )
    if settings.embedding_provider not in {
        "sentence-transformers", "ollama", "openai"
    }:
        raise ValueError(
            "EMBEDDING_PROVIDER must be 'sentence-transformers', 'ollama' or 'openai'."
        )
