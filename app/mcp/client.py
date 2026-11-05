"""MCP Client：stdio 连接、工具发现和 LangChain 工具加载。"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def server_params() -> StdioServerParameters:
    """启动 app.mcp.server 的参数。

    env 继承当前进程，确保子进程能读到 .env 里的 API Key；
    cwd 指向项目根目录，确保 data/chroma 和 .env 路径正确。
    """
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp.server"],
        cwd=str(PROJECT_ROOT),
        env=os.environ.copy(),
    )


def connection_config() -> dict[str, Any]:
    """langchain-mcp-adapters 使用的 stdio 连接配置。"""
    return {
        "knowledge": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "app.mcp.server"],
            "cwd": str(PROJECT_ROOT),
            "env": os.environ.copy(),
        }
    }


@asynccontextmanager
async def open_session() -> AsyncIterator[ClientSession]:
    """打开一个原始 MCP ClientSession，用于工具发现和直接调用。"""
    async with stdio_client(server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def list_tools() -> list[Any]:
    """通过 MCP 协议获取 Server 的工具列表。"""
    async with open_session() as session:
        result = await session.list_tools()
        return list(result.tools)


async def call_tool(name: str, arguments: dict[str, Any] | None = None) -> str:
    """直接调用 MCP 工具，返回文本结果。"""
    async with open_session() as session:
        result = await session.call_tool(name, arguments or {})
        parts: list[str] = []
        for item in result.content:
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(str(text))
        return "".join(parts)


async def load_langchain_tools() -> list[BaseTool]:
    """通过适配器把 MCP 工具转换成 LangChain Tool，供 Agent 使用。"""
    client = MultiServerMCPClient(connection_config())
    return await client.get_tools(server_name="knowledge")


if __name__ == "__main__":
    async def _demo() -> None:
        tools = await list_tools()
        print("MCP tools:")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
        print("\nlist_documents:")
        print(await call_tool("list_documents"))
        print("\nsearch_documents:")
        print(
            await call_tool(
                "search_documents",
                {"query": "Agentic RAG 最多重试几次", "k": 2},
            )
        )

    asyncio.run(_demo())
