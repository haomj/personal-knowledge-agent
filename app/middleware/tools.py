"""Middleware 演示工具：只读检索、高风险保存、可重试查询。"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass

from langchain_core.tools import tool

from app.hitl.report import save_research_report as save_report_file
from app.rag.retriever import build_default_retriever


@dataclass(frozen=True)
class UserContext:
    """运行时上下文：权限中间件从这里读取用户角色。"""

    user_id: str
    role: str = "viewer"


@tool
def read_knowledge(query: str) -> str:
    """只读检索本地知识库，返回命中的文档片段和来源。"""
    documents = build_default_retriever().retrieve(query)
    results = [
        {
            "source": document.metadata.get("source", ""),
            "score": round(float(document.metadata.get("score", 0.0)), 4),
            "content": document.page_content[:800],
        }
        for document in documents
    ]
    return json.dumps(
        {"query": query, "count": len(results), "results": results},
        ensure_ascii=False,
    )


@tool
def save_research_report(title: str, content: str) -> str:
    """保存正式研究报告（高风险写操作，仅管理员可调用）。"""
    path = save_report_file(title, content)
    return f"报告已保存: {path}"


_FAILURES: dict[str, int] = {}
_FAILURE_LOCK = threading.Lock()


@tool
def lookup_with_retry(query: str) -> str:
    """模拟不稳定的只读查询：前两次失败，第三次成功，用于演示重试中间件。"""
    with _FAILURE_LOCK:
        count = _FAILURES.get(query, 0) + 1
        _FAILURES[query] = count
    if count <= 2:
        raise RuntimeError(f"临时网络错误（第 {count} 次）")
    return f"查询成功（第 {count} 次）: {query}"
