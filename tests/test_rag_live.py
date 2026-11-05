"""Agentic RAG 的真实端到端测试：默认跳过。"""

from __future__ import annotations

import os

import pytest

from app.config import get_embeddings
from app.rag.graph import build_rag_graph, initial_rag_state
from app.rag.loader import load_documents
from app.rag.retriever import KnowledgeRetriever
from app.rag.splitter import split_documents
from app.rag.vector_store import build_vector_store

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1",
    reason="需要真实 Embedding 和模型 API；设置 RUN_LIVE_TESTS=1 运行",
)


def test_rag_end_to_end_with_real_embeddings_and_model(tmp_path):
    documents = load_documents("data/docs")
    chunks = split_documents(documents)
    store = build_vector_store(
        chunks,
        persist_directory=tmp_path / "chroma",
        collection_name="live_test",
        embeddings=get_embeddings(),
        reset=True,
    )
    retriever = KnowledgeRetriever(store, k=4)
    graph = build_rag_graph(
        retriever=retriever,
        system_prompt="回答尽量简短，并标注来源。",
        max_retries=1,
    )

    result = graph.invoke(
        initial_rag_state("Agentic RAG 在检索结果不足时最多重试几次？")
    )

    assert result["retrieval_attempts"] >= 1
    assert result["sources"]
    sources = [item["source"] for item in result["sources"]]
    assert "data/docs/03-agentic-rag.md" in sources
    assert result["messages"][-1].content
