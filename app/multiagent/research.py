"""Research Agent：通过 Tavily 联网搜索，并总结带 URL 的答案。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from langchain_core.language_models import BaseChatModel
from langchain_tavily import TavilySearch

from app.config import get_model
from app.multiagent.models import SubagentResult
from app.text import content_text

logger = logging.getLogger(__name__)

SearchFn = Callable[[str, int], Awaitable[list[dict[str, Any]]]]


async def tavily_search(query: str, max_results: int) -> list[dict[str, Any]]:
    """默认联网检索：Tavily 返回 dict，其中 results 是结果列表。"""
    tool = TavilySearch(
        max_results=max_results,
        topic="general",
        include_answer=False,
    )
    raw = await asyncio.to_thread(tool.invoke, {"query": query})
    if isinstance(raw, dict):
        return list(raw.get("results", []))
    if isinstance(raw, list):
        return raw
    return []


def _format_results(results: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for index, result in enumerate(results, 1):
        title = result.get("title", "")
        url = result.get("url", "")
        content = str(result.get("content", "")).strip()
        if len(content) > 1200:
            content = content[:1200] + "..."
        parts.append(
            f"[{index}] 标题: {title}\nURL: {url}\n{content}"
        )
    return "\n\n".join(parts)


async def run_research_agent(
    query: str,
    *,
    search: SearchFn | None = None,
    model: BaseChatModel | None = None,
    max_results: int = 5,
) -> SubagentResult:
    """只拿子问题和联网结果，不接触 Supervisor 的完整对话。"""
    search_fn = search or tavily_search
    chat_model = model or get_model(streaming=False)

    try:
        results = await search_fn(query, max_results)
    except Exception as exc:
        logger.warning("research_agent.search failed: %s", exc)
        return SubagentResult(
            agent="research",
            query=query,
            answer=f"联网检索失败：{exc}",
            sources=[],
        )

    if not results:
        return SubagentResult(
            agent="research",
            query=query,
            answer="没有找到相关的联网搜索结果。",
            sources=[],
        )

    prompt = (
        "你是 Research Agent，只负责联网资料。\n"
        "请只根据下面的搜索结果回答子问题；信息不足时明确说明。\n"
        "关键结论后标注 URL；不要编造未出现在结果中的事实。\n\n"
        f"子问题：{query}\n\n"
        f"搜索结果：\n{_format_results(results)}"
    )
    response = await asyncio.to_thread(chat_model.invoke, prompt)
    answer = content_text(response.content)

    sources = sorted(
        {
            str(result.get("url", ""))
            for result in results
            if result.get("url")
        }
    )
    logger.info(
        "research_agent.done query=%r results=%d sources=%d",
        query,
        len(results),
        len(sources),
    )
    return SubagentResult(
        agent="research",
        query=query,
        answer=answer,
        sources=sources,
    )
