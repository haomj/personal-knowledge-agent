"""Knowledge Agent：通过 MCP 检索本地知识库，并总结答案。"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable

from langchain_core.language_models import BaseChatModel

from app.config import get_model
from app.mcp.client import call_tool
from app.multiagent.models import SubagentResult
from app.text import content_text

logger = logging.getLogger(__name__)

SearchFn = Callable[[str, int], Awaitable[list[dict[str, Any]]]]


async def mcp_search(query: str, k: int) -> list[dict[str, Any]]:
    """默认检索：复用 Day 4 的 MCP search_documents 工具。"""
    text = await call_tool(
        "search_documents", {"query": query, "k": k}
    )
    data = json.loads(text)
    if "error" in data:
        raise RuntimeError(str(data["error"]))
    return list(data.get("results", []))


def _format_results(results: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for index, result in enumerate(results, 1):
        source = result.get("source", "unknown")
        content = str(result.get("content", "")).strip()
        if len(content) > 1200:
            content = content[:1200] + "..."
        parts.append(f"[{index}] 来源: {source}\n{content}")
    return "\n\n".join(parts)


async def run_knowledge_agent(
    query: str,
    *,
    search: SearchFn | None = None,
    model: BaseChatModel | None = None,
    top_k: int = 4,
) -> SubagentResult:
    """只拿子问题和检索结果，不接触 Supervisor 的完整对话。"""
    search_fn = search or mcp_search
    chat_model = model or get_model(streaming=False)

    try:
        results = await search_fn(query, top_k)
    except Exception as exc:
        logger.warning("knowledge_agent.search failed: %s", exc)
        return SubagentResult(
            agent="knowledge",
            query=query,
            answer=f"本地知识库检索失败：{exc}",
            sources=[],
        )

    if not results:
        return SubagentResult(
            agent="knowledge",
            query=query,
            answer="本地知识库没有检索到相关资料。",
            sources=[],
        )

    prompt = (
        "你是 Knowledge Agent，只负责本地知识库。\n"
        "请只根据下面的资料回答子问题；资料不足时明确说明。\n"
        "关键结论后使用 [来源: 文件路径] 标注出处。\n\n"
        f"子问题：{query}\n\n"
        f"检索资料：\n{_format_results(results)}"
    )
    response = await asyncio.to_thread(chat_model.invoke, prompt)
    answer = content_text(response.content)

    sources = sorted(
        {
            str(result.get("source", ""))
            for result in results
            if result.get("source")
        }
    )
    logger.info(
        "knowledge_agent.done query=%r docs=%d sources=%d",
        query,
        len(results),
        len(sources),
    )
    return SubagentResult(
        agent="knowledge",
        query=query,
        answer=answer,
        sources=sources,
    )
