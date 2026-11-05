"""Chroma 向量库：创建、重建和统计。"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# 文本检索使用余弦距离；Chroma 在创建集合时固定该配置。
COLLECTION_METADATA = {"hnsw:space": "cosine"}


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
    """把 chunks 写入 Chroma。

    reset=True 时会删除旧目录，避免重复写入产生重复 chunk。
    """
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


def collection_count(store: Chroma) -> int:
    """返回集合中的向量数量；不同 Chroma 版本内部字段可能不同。"""
    collection = getattr(store, "_collection", None)
    if collection is None:
        return 0
    return int(collection.count())
