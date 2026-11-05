"""MCP Server：把 Day 2 的 Retriever 暴露成两个工具。

工具：
- list_documents()：列出本地知识库文档；
- search_documents(query, k=4)：复用 Agentic RAG 的 Chroma 检索。

第一版使用 stdio transport，由 MCP Client 作为子进程启动。
注意：stdio 模式下 stdout 属于 JSON-RPC 协议，业务日志必须走 stderr。
"""

from __future__ import annotations

import json
import logging
import os

from mcp.server.fastmcp import FastMCP

from app.config import configure_logging, get_settings
from app.rag.loader import load_documents
from app.rag.retriever import KnowledgeRetriever, build_default_retriever

logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="personal-knowledge-base",
    instructions="本地技术知识库：可以列出文档、按语义检索文档片段。",
    log_level="WARNING",
)

_retriever: KnowledgeRetriever | None = None


def _get_retriever(k: int) -> KnowledgeRetriever:
    global _retriever
    retriever = _retriever
    if retriever is None:
        retriever = build_default_retriever(top_k=k)
        _retriever = retriever
    retriever.k = max(1, min(k, 10))
    return retriever


@mcp.tool()
def list_documents() -> str:
    """列出本地知识库中的 Markdown/TXT 文档及其来源路径。"""
    try:
        documents = load_documents(get_settings().docs_dir)
        items = [
            {
                "source": document.metadata.get("source", ""),
                "relative_source": document.metadata.get(
                    "relative_source", ""
                ),
                "title": document.metadata.get("title", ""),
            }
            for document in documents
        ]
        logger.info("mcp.tool.list_documents count=%d", len(items))
        return json.dumps(
            {"count": len(items), "documents": items},
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.exception("mcp.tool.list_documents failed")
        return json.dumps(
            {"error": f"列出文档失败: {exc}"}, ensure_ascii=False
        )


@mcp.tool()
def search_documents(query: str, k: int = 4) -> str:
    """在本地知识库中检索与 query 相关的文档片段，返回来源和分数。"""
    if not query or not query.strip():
        return json.dumps(
            {"error": "query 不能为空"}, ensure_ascii=False
        )
    try:
        documents = _get_retriever(k).retrieve(query)
        results = [
            {
                "source": document.metadata.get("source", ""),
                "title": document.metadata.get("title", ""),
                "score": round(float(document.metadata.get("score", 0.0)), 4),
                "content": document.page_content[:1200],
            }
            for document in documents
        ]
        logger.info(
            "mcp.tool.search_documents query=%r results=%d",
            query,
            len(results),
        )
        return json.dumps(
            {"query": query, "count": len(results), "results": results},
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.exception("mcp.tool.search_documents failed")
        return json.dumps(
            {"error": f"检索失败: {exc}"}, ensure_ascii=False
        )


def main() -> None:
    # stdio 模式下 stdout 是 JSON-RPC 通道；日志默认只输出 WARNING，
    # 需要排查时设置 MCP_LOG_LEVEL=INFO。
    configure_logging(os.getenv("MCP_LOG_LEVEL", "WARNING"))
    logger.info("mcp.server starting transport=stdio")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
