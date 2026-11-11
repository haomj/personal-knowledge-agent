"""Milvus 向量库：Milvus Lite（本地文件）或远程 Milvus Server。"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus

from app.config import Settings

logger = logging.getLogger(__name__)


def _connection_args(settings: Settings) -> dict[str, str]:
    args = {"uri": settings.milvus_uri}
    if settings.milvus_token:
        args["token"] = settings.milvus_token
    return args


def _ensure_local_parent(uri: str) -> None:
    """Milvus Lite 使用本地文件；远程 URI 不需要创建目录。"""
    if uri.startswith(("http://", "https://", "tcp://", ":memory:")):
        return
    Path(uri).expanduser().parent.mkdir(parents=True, exist_ok=True)


def _index_params(settings: Settings) -> dict:
    return {
        "index_type": settings.milvus_index_type,
        "metric_type": settings.milvus_metric_type,
        "params": {},
    }


def get_milvus_store(
    *,
    embeddings: Embeddings,
    settings: Settings,
) -> Milvus:
    """打开一个 Milvus 集合；集合不存在时创建空集合。"""
    _ensure_local_parent(settings.milvus_uri)
    return Milvus(
        embedding_function=embeddings,
        collection_name=settings.milvus_collection,
        connection_args=_connection_args(settings),
        index_params=_index_params(settings),
    )


def build_milvus_store(
    documents: list[Document],
    *,
    embeddings: Embeddings,
    settings: Settings,
    reset: bool = True,
) -> Milvus:
    """把 chunks 写入 Milvus；reset=True 时删除旧集合。"""
    _ensure_local_parent(settings.milvus_uri)
    store = Milvus.from_documents(
        documents=documents,
        embedding=embeddings,
        connection_args=_connection_args(settings),
        collection_name=settings.milvus_collection,
        index_params=_index_params(settings),
        drop_old=reset,
    )
    logger.info(
        "milvus.indexed chunks=%d collection=%s uri=%s",
        len(documents),
        settings.milvus_collection,
        settings.milvus_uri,
    )
    return store


def milvus_count(store: Milvus) -> int:
    """用 Milvus 的 collection stats 获取行数，兼容 Milvus Lite。"""
    client = getattr(store, "client", None)
    collection_name = getattr(store, "collection_name", None)
    if client is not None and collection_name:
        try:
            stats = client.get_collection_stats(collection_name=collection_name)
            return int(stats.get("row_count", 0))
        except Exception:
            pass

    col = getattr(store, "col", None)
    num_entities = getattr(col, "num_entities", None)
    if num_entities is not None:
        try:
            return int(num_entities)
        except Exception:
            pass
    return 0
