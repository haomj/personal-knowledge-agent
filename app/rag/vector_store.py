"""向量库后端：默认 Chroma，可通过 VECTOR_STORE=milvus 切换。

- Chroma：本地持久化目录，默认后端。
- Milvus：Milvus Lite 本地文件（MILVUS_URI=data/milvus.db）或远程 Server
  （MILVUS_URI=http://localhost:19530）。
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# 文本检索使用余弦距离；Chroma 在创建集合时固定该配置。
COLLECTION_METADATA = {"hnsw:space": "cosine"}


def _validate_backend(backend: str) -> str:
    normalized = backend.lower().strip()
    if normalized not in {"chroma", "milvus"}:
        raise ValueError(
            f"不支持的 VECTOR_STORE={backend!r}，可选值：chroma、milvus"
        )
    return normalized


# ---------------------------------------------------------------------------
# Chroma 低层 API（保留给现有测试与直接使用）
# ---------------------------------------------------------------------------


def get_vector_store(
    *,
    persist_directory: str | Path,
    collection_name: str,
    embeddings: Embeddings,
) -> Chroma:
    """打开或创建一个 Chroma 集合。"""
    path = Path(persist_directory)
    path.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(path),
        collection_metadata=COLLECTION_METADATA,
    )


def build_vector_store(
    documents: list[Document],
    *,
    persist_directory: str | Path,
    collection_name: str,
    embeddings: Embeddings,
    reset: bool = True,
) -> Chroma:
    """把 chunks 写入 Chroma；reset=True 时删除旧目录。"""
    path = Path(persist_directory)
    if reset and path.exists():
        shutil.rmtree(path)
        logger.info("chroma.reset directory=%s", path)
    path.mkdir(parents=True, exist_ok=True)

    store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(path),
        collection_name=collection_name,
        collection_metadata=COLLECTION_METADATA,
    )
    logger.info(
        "chroma.indexed chunks=%d collection=%s dir=%s",
        len(documents),
        collection_name,
        path,
    )
    return store


# ---------------------------------------------------------------------------
# 可切换后端
# ---------------------------------------------------------------------------


def get_configured_vector_store(
    *,
    embeddings: Embeddings,
    settings: Settings | None = None,
) -> VectorStore:
    """按 VECTOR_STORE 打开默认集合。"""
    settings = settings or get_settings()
    backend = _validate_backend(settings.vector_store)

    if backend == "milvus":
        # 懒加载：Chroma 模式不 import pymilvus，避免它的全局 URI 配置影响启动。
        from app.rag.milvus_store import get_milvus_store

        logger.info(
            "vector_store.backend=milvus collection=%s uri=%s",
            settings.milvus_collection,
            settings.milvus_uri,
        )
        return get_milvus_store(embeddings=embeddings, settings=settings)

    logger.info(
        "vector_store.backend=chroma collection=%s dir=%s",
        settings.chroma_collection,
        settings.chroma_dir,
    )
    return get_vector_store(
        persist_directory=settings.chroma_dir,
        collection_name=settings.chroma_collection,
        embeddings=embeddings,
    )


def build_configured_vector_store(
    documents: list[Document],
    *,
    embeddings: Embeddings,
    settings: Settings | None = None,
    reset: bool = True,
) -> VectorStore:
    """按 VECTOR_STORE 重建/追加索引。"""
    settings = settings or get_settings()
    backend = _validate_backend(settings.vector_store)

    if backend == "milvus":
        from app.rag.milvus_store import build_milvus_store

        return build_milvus_store(
            documents,
            embeddings=embeddings,
            settings=settings,
            reset=reset,
        )

    return build_vector_store(
        documents,
        persist_directory=settings.chroma_dir,
        collection_name=settings.chroma_collection,
        embeddings=embeddings,
        reset=reset,
    )


def collection_count(store: Any) -> int:
    """返回集合中的向量数量，兼容 Chroma 与 Milvus。"""
    chroma_collection = getattr(store, "_collection", None)
    if chroma_collection is not None:
        return int(chroma_collection.count())
    if getattr(store, "col", None) is not None:
        from app.rag.milvus_store import milvus_count

        return milvus_count(store)
    return 0
