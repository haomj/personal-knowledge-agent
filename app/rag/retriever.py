"""知识库检索器：封装 Chroma 的相似度检索。"""

from __future__ import annotations

import logging
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import get_embeddings, get_settings
from app.rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """把 Chroma 检索结果整理成带 score 的 Document 列表。"""

    def __init__(self, store: Chroma, *, k: int = 4):
        self.store = store
        self.k = k

    def retrieve(self, query: str) -> list[Document]:
        if not query.strip():
            return []
        pairs = self.store.similarity_search_with_relevance_scores(
            query, k=self.k
        )
        results: list[Document] = []
        for document, score in pairs:
            results.append(
                Document(
                    page_content=document.page_content,
                    metadata={
                        **document.metadata,
                        "score": float(score),
                    },
                )
            )
        top_score: float | None = (
            float(results[0].metadata["score"]) if results else None
        )
        logger.info(
            "rag.retrieve query=%r docs=%d top_score=%s",
            query,
            len(results),
            f"{top_score:.3f}" if top_score is not None else "none",
        )
        return results


def build_default_retriever(*, top_k: int | None = None) -> KnowledgeRetriever:
    """按 .env 配置构建默认检索器。"""
    settings = get_settings()
    store = get_vector_store(
        persist_directory=settings.chroma_dir,
        collection_name=settings.chroma_collection,
        embeddings=get_embeddings(),
    )
    return KnowledgeRetriever(store, k=top_k or settings.rag_top_k)


def sources_from_documents(documents: list[Document]) -> list[dict[str, Any]]:
    """按来源去重，保留每个来源的最高分。"""
    best: dict[str, dict[str, Any]] = {}
    for document in documents:
        source = str(document.metadata.get("source", "unknown"))
        score = float(document.metadata.get("score", 0.0))
        current = best.get(source)
        if current is None or score > current["score"]:
            best[source] = {
                "source": source,
                "title": document.metadata.get("title", ""),
                "score": score,
            }
    return sorted(best.values(), key=lambda item: item["score"], reverse=True)
