"""构建本地知识库索引。

用法：

    uv run python -m app.rag.ingest
    uv run python -m app.rag.ingest --append
    uv run python -m app.rag.ingest --docs-dir data/docs --persist-dir data/chroma
"""

from __future__ import annotations

import argparse
import logging

from app.config import configure_logging, get_embeddings, get_settings
from app.rag.loader import load_documents
from app.rag.splitter import split_documents
from app.rag.vector_store import build_vector_store

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="构建 Agentic RAG 知识库索引")
    parser.add_argument("--docs-dir", default=settings.docs_dir)
    parser.add_argument("--persist-dir", default=settings.chroma_dir)
    parser.add_argument("--collection", default=settings.chroma_collection)
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

    documents = load_documents(args.docs_dir)
    chunks = split_documents(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    logger.info(
        "ingest.loaded documents=%d chunks=%d", len(documents), len(chunks)
    )

    build_vector_store(
        chunks,
        persist_directory=args.persist_dir,
        collection_name=args.collection,
        embeddings=get_embeddings(),
        reset=not args.append,
    )

    print(
        f"索引完成：documents={len(documents)} chunks={len(chunks)} "
        f"collection={args.collection} dir={args.persist_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
