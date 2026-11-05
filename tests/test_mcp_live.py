"""MCP 真实端到端测试：默认跳过。"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from app.mcp.agent import ask_mcp_once, build_mcp_agent
from app.mcp.client import call_tool, load_langchain_tools

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1"
    or not Path("data/chroma").exists(),
    reason="需要真实 Embedding/模型 API 和已构建的 Chroma 索引",
)


def test_mcp_search_tool_with_real_embeddings():
    async def _run() -> dict:
        text = await call_tool(
            "search_documents",
            {"query": "Agentic RAG 检索不足时最多重试几次", "k": 2},
        )
        return json.loads(text)

    data = asyncio.run(_run())
    assert data["count"] >= 1
    assert data["results"][0]["source"].startswith("data/docs/")


def test_mcp_agent_calls_tools_end_to_end():
    async def _run() -> str:
        tools = await load_langchain_tools()
        agent = build_mcp_agent(tools)
        return await ask_mcp_once(
            agent,
            "MCP 的 stdio 适合什么场景？",
            "mcp-live-agent",
        )

    answer = asyncio.run(_run())
    assert "stdio" in answer.lower()
    assert "data/docs/04-mcp.md" in answer
