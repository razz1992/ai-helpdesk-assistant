"""Load Markdown KB articles and build an in-memory FAISS retriever."""
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import KNOWLEDGE_BASE_DIR
from src.langchain_components.llm import get_embeddings


def load_knowledge_documents() -> list[Document]:
    documents: list[Document] = []
    for path in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        # Keep article metadata explicit and simple for the first learning pass.
        lines = text.splitlines()
        metadata = {"source": path.name}
        for line in lines[:12]:
            if line.startswith("document_id:"):
                metadata["document_id"] = line.split(":", 1)[1].strip()
            elif line.startswith("title:"):
                metadata["title"] = line.split(":", 1)[1].strip()
            elif line.startswith("category:"):
                metadata["category"] = line.split(":", 1)[1].strip()
        body = text.split("---", 2)[-1].strip() if text.startswith("---") else text
        documents.append(Document(page_content=body, metadata=metadata))
    if not documents:
        raise FileNotFoundError(f"No Markdown KB files found in {KNOWLEDGE_BASE_DIR}")
    return documents


@lru_cache(maxsize=1)
def get_retriever():
    """Build once per process. Embedding calls occur on first use."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=100, add_start_index=True
    )
    chunks = splitter.split_documents(load_knowledge_documents())
    vector_store = FAISS.from_documents(chunks, get_embeddings())
    return vector_store.as_retriever(search_kwargs={"k": 3})
