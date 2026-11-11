"""向量库后端切换测试：Chroma 与 Milvus Lite。"""

from __future__ import annotations

from dataclasses import replace

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import get_settings
from app.rag.retriever import KnowledgeRetriever
from app.rag.vector_store import (
    build_configured_vector_store,
    collection_count,
    get_configured_vector_store,
)


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [
            1.0 if ("Chroma" in text or "cosine" in text) else 0.0,
            1.0 if ("Milvus" in text or "向量" in text) else 0.0,
        ]


def _documents() -> list[Document]:
    return [
        Document(
            page_content="Milvus 是向量数据库",
            metadata={"source": "milvus.md", "title": "Milvus"},
        ),
        Document(
            page_content="Chroma 文本检索使用 cosine 距离",
            metadata={"source": "chroma.md", "title": "Chroma"},
        ),
    ]


def test_chroma_backend_roundtrip(tmp_path):
    settings = replace(
        get_settings(),
        vector_store="chroma",
        chroma_dir=str(tmp_path / "chroma"),
        chroma_collection="test_chroma",
    )
    store = build_configured_vector_store(
        _documents(),
        embeddings=FakeEmbeddings(),
        settings=settings,
        reset=True,
    )
    assert type(store).__name__ == "Chroma"
    assert collection_count(store) == 2

    results = KnowledgeRetriever(store, k=1).retrieve("Chroma cosine")
    assert results[0].metadata["source"] == "chroma.md"


def test_milvus_backend_roundtrip(tmp_path):
    settings = replace(
        get_settings(),
        vector_store="milvus",
        milvus_uri=str(tmp_path / "milvus.db"),
        milvus_collection="test_milvus",
        milvus_index_type="AUTOINDEX",
        milvus_metric_type="COSINE",
    )
    store = build_configured_vector_store(
        _documents(),
        embeddings=FakeEmbeddings(),
        settings=settings,
        reset=True,
    )
    assert type(store).__name__ == "Milvus"
    assert collection_count(store) == 2

    results = KnowledgeRetriever(store, k=1).retrieve("Milvus 向量数据库")
    assert results[0].metadata["source"] == "milvus.md"


def test_empty_milvus_index_returns_empty(tmp_path):
    settings = replace(
        get_settings(),
        vector_store="milvus",
        milvus_uri=str(tmp_path / "empty.db"),
        milvus_collection="empty_collection",
    )
    store = get_configured_vector_store(
        embeddings=FakeEmbeddings(), settings=settings
    )
    retriever = KnowledgeRetriever(store, k=2)

    assert retriever.retrieve("anything") == []


def test_invalid_backend_is_rejected(tmp_path):
    settings = replace(
        get_settings(),
        vector_store="weaviate",
        milvus_uri=str(tmp_path / "milvus.db"),
    )
    with pytest.raises(ValueError, match="不支持的 VECTOR_STORE"):
        get_configured_vector_store(
            embeddings=FakeEmbeddings(), settings=settings
        )
