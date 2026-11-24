"""Google Gemini and optional chat provider factories, plus local embeddings."""
import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from src.config import settings, validate_provider_config


def get_chat_model() -> BaseChatModel:
    """Return the configured chat model."""
    validate_provider_config()

    if settings.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # This Gemini model uses fixed sampling defaults; omit temperature.
        return ChatGoogleGenerativeAI(
            model=settings.google_model,
            google_api_key=settings.google_api_key,
        )

    if settings.provider == "deepseek":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=settings.temperature,
        )

    if settings.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=settings.temperature,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=os.getenv("OLLAMA_CHAT_MODEL", "llama3.2"),
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
    )


def get_embeddings() -> Embeddings:
    """Return embeddings independently of the configured chat provider."""
    validate_provider_config()

    if settings.embedding_provider == "sentence-transformers":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=settings.sentence_transformer_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )

    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )
