"""RAG 离线测试：加载、切分、Chroma 检索，不访问网络。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.rag.loader import load_documents
from app.rag.retriever import KnowledgeRetriever, sources_from_documents
from app.rag.splitter import split_documents
from app.rag.vector_store import build_vector_store, collection_count


class FakeEmbeddings(Embeddings):
    """测试用的确定性二维关键词向量，避免调用真实 Embedding API。

    维度 0 表示 LangGraph/compile，维度 1 表示 Chroma/cosine/距离。
    这不是语义模型，只用于验证 Chroma 的写入和检索链路。
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        langgraph = 1.0 if ("LangGraph" in text or "compile" in text) else 0.0
        chroma = (
            1.0
            if ("Chroma" in text or "cosine" in text or "距离" in text)
            else 0.0
        )
        return [langgraph, chroma]


def test_load_corpus_and_split():
    documents = load_documents("data/docs")
    assert len(documents) >= 10

    chunks = split_documents(documents)
    assert len(chunks) > len(documents)
    for chunk in chunks:
        assert chunk.metadata["source"]
        assert chunk.metadata["chunk_id"]
        assert "chunk_index" in chunk.metadata


def test_chroma_roundtrip_with_fake_embeddings(tmp_path: Path):
    documents = [
        Document(
            page_content="LangGraph 的 compile 会返回 CompiledStateGraph",
            metadata={"source": "a.md", "title": "LangGraph"},
        ),
        Document(
            page_content="Chroma 文本检索通常使用 cosine 余弦距离",
            metadata={"source": "b.md", "title": "Chroma"},
        ),
    ]
    store = build_vector_store(
        documents,
        persist_directory=tmp_path / "chroma",
        collection_name="test_collection",
        embeddings=FakeEmbeddings(),
        reset=True,
    )
    assert collection_count(store) == 2

    retriever = KnowledgeRetriever(store, k=1)
    results = retriever.retrieve("Chroma cosine 距离")
    assert len(results) == 1
    assert results[0].metadata["source"] == "b.md"
    assert "score" in results[0].metadata


def test_sources_from_documents_deduplicates_by_source():
    documents = [
        Document("a", metadata={"source": "a.md", "score": 0.5}),
        Document("b", metadata={"source": "a.md", "score": 0.8}),
        Document("c", metadata={"source": "b.md", "score": 0.6}),
    ]
    sources = sources_from_documents(documents)
    assert [item["source"] for item in sources] == ["a.md", "b.md"]
    assert sources[0]["score"] == 0.8
