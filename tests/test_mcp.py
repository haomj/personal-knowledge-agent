"""MCP 离线测试：工具发现和 list_documents，不访问模型 API。"""

from __future__ import annotations

import asyncio
import json

from app.mcp.client import call_tool, list_tools, load_langchain_tools


def test_mcp_tool_discovery():
    tools = asyncio.run(list_tools())
    names = {tool.name for tool in tools}
    assert names == {"list_documents", "search_documents"}
    for tool in tools:
        assert tool.description


def test_mcp_list_documents():
    text = asyncio.run(call_tool("list_documents"))
    data = json.loads(text)
    assert data["count"] >= 10
    assert any(
        item["source"] == "data/docs/03-agentic-rag.md"
        for item in data["documents"]
    )


def test_langchain_adapter_loads_tools():
    tools = asyncio.run(load_langchain_tools())
    names = {tool.name for tool in tools}
    assert names == {"list_documents", "search_documents"}
