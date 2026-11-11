"""构建本地知识库索引。

用法：

    uv run python -m app.rag.ingest
    uv run python -m app.rag.ingest --backend milvus
    uv run python -m app.rag.ingest --backend milvus --milvus-uri data/milvus.db
    uv run python -m app.rag.ingest --append

默认后端由 VECTOR_STORE 决定：chroma 或 milvus。
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace

from app.config import configure_logging, get_embeddings, get_settings
from app.rag.loader import load_documents
from app.rag.splitter import split_documents
from app.rag.vector_store import build_configured_vector_store

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="构建 Agentic RAG 知识库索引")
    parser.add_argument("--docs-dir", default=settings.docs_dir)
    parser.add_argument(
        "--backend",
        choices=["chroma", "milvus"],
        default=settings.vector_store,
        help="向量库后端，默认读取 VECTOR_STORE",
    )
    parser.add_argument(
        "--persist-dir",
        default=settings.chroma_dir,
        help="Chroma 持久化目录",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="集合名；不填则使用对应后端的默认集合",
    )
    parser.add_argument(
        "--milvus-uri",
        default=settings.milvus_uri,
        help="Milvus Lite 文件路径或远程 Milvus URI",
    )
    parser.add_argument(
        "--milvus-collection",
        default=None,
        help="Milvus 集合名，默认 MILVUS_COLLECTION",
    )
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument(
        "--append",
        action="store_true",
        help="不删除旧索引，直接追加（默认会重建）",
    )
    parser.add_argument("--log-level")
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    chroma_collection = args.collection or settings.chroma_collection
    milvus_collection = (
        args.milvus_collection
        or args.collection
        or settings.milvus_collection
    )
    effective = replace(
        settings,
        vector_store=args.backend,
        chroma_dir=args.persist_dir,
        chroma_collection=chroma_collection,
        milvus_uri=args.milvus_uri,
        milvus_collection=milvus_collection,
    )

    documents = load_documents(args.docs_dir)
    chunks = split_documents(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    logger.info(
        "ingest.loaded documents=%d chunks=%d backend=%s",
        len(documents),
        len(chunks),
        args.backend,
    )

    build_configured_vector_store(
        chunks,
        embeddings=get_embeddings(),
        settings=effective,
        reset=not args.append,
    )

    if args.backend == "milvus":
        location = f"uri={args.milvus_uri} collection={milvus_collection}"
    else:
        location = f"dir={args.persist_dir} collection={chroma_collection}"
    print(
        f"索引完成：documents={len(documents)} chunks={len(chunks)} "
        f"backend={args.backend} {location}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
