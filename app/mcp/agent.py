"""MCP Agent：通过 MCP Client 加载知识库工具，让 Agent 自主调用。

使用流程：

    MCP Client -> stdio -> MCP Server -> search_documents / list_documents

Agent 使用 LangChain 的 ``create_agent``，工具由 MCP Server 动态发现。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.config import get_model, get_settings
from app.mcp.client import load_langchain_tools
from app.memory.store import (
    load_preferences,
    preference_prompt,
    save_preference,
)
from app.text import content_text

logger = logging.getLogger(__name__)

MCP_INSTRUCTIONS = """你可以通过 MCP 调用本地知识库工具：
- list_documents：列出知识库中的文档；
- search_documents：按语义检索文档片段。

回答本地知识库问题时，必须先调用 search_documents，不要凭记忆编造。
引用资料时使用 [来源: 文件路径]；如果检索结果不足，明确说明。"""


def _system_prompt(user_id: str, store: Any) -> str:
    parts = [get_settings().system_prompt, MCP_INSTRUCTIONS]
    preferences = preference_prompt(store, user_id)
    if preferences:
        parts.append(preferences)
    return "\n\n".join(part for part in parts if part)


def build_mcp_agent(
    tools: list[BaseTool],
    *,
    checkpointer: Any = None,
    store: Any = None,
    user_id: str = "default",
):
    return create_agent(
        get_model(),
        tools=tools,
        system_prompt=_system_prompt(user_id, store),
        checkpointer=checkpointer,
        store=store,
    )


def _print_tool_trace(messages: list[Any]) -> None:
    """打印 Agent 的工具调用和工具返回，便于观察 MCP 是否真的被调用。"""
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                print(
                    f"[MCP] 调用 {call.get('name')} "
                    f"{call.get('args', {})}"
                )
        elif isinstance(message, ToolMessage):
            text = content_text(message.content)
            preview = text[:300] + ("..." if len(text) > 300 else "")
            print(f"[MCP] {message.name or 'tool'} 返回: {preview}")


def _current_turn_messages(messages: list[Any]) -> list[Any]:
    """只取最后一轮 HumanMessage 之后的消息，避免重复打印历史工具调用。"""
    last_human = None
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            last_human = index
    if last_human is None:
        return messages
    return messages[last_human:]


async def ask_mcp_stream(
    agent: Any, question: str, thread_id: str
) -> str:
    """流式执行 MCP Agent，打印 token，结束后打印工具调用记录。"""
    config = {"configurable": {"thread_id": thread_id}}
    chunks: list[str] = []

    async for chunk, _metadata in agent.astream(
        {"messages": [HumanMessage(content=question)]},
        config=config,
        stream_mode="messages",
    ):
        if not isinstance(chunk, AIMessage):
            continue
        text = content_text(chunk.content)
        if text:
            print(text, end="", flush=True)
            chunks.append(text)

    print()
    snapshot = await agent.aget_state(config)
    if snapshot:
        _print_tool_trace(
            _current_turn_messages(snapshot.values.get("messages", []))
        )
    return "".join(chunks)


async def ask_mcp_once(
    agent: Any, question: str, thread_id: str
) -> str:
    """非流式执行 MCP Agent，先打印工具调用记录，再打印最终答案。"""
    config = {"configurable": {"thread_id": thread_id}}
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )
    messages = result["messages"]
    _print_tool_trace(_current_turn_messages(messages))
    answer = content_text(messages[-1].content)
    print(answer)
    return answer


async def run_mcp_cli(
    *,
    question: str | None,
    thread_id: str,
    user_id: str,
    store: Any,
    checkpointer: Any,
    stream: bool = True,
) -> int:
    """MCP 模式的交互循环；由 app.cli 的 --mode mcp 调用。"""
    tools = await load_langchain_tools()
    agent = build_mcp_agent(
        tools,
        checkpointer=checkpointer,
        store=store,
        user_id=user_id,
    )

    async def ask(text: str) -> str:
        if stream:
            return await ask_mcp_stream(agent, text, thread_id)
        return await ask_mcp_once(agent, text, thread_id)

    if question:
        logger.info(
            "cli.mcp.once user=%s thread=%s", user_id, thread_id
        )
        await ask(question)
        return 0

    print("个人知识研究助手 v0.4 MCP Agent")
    print(f"当前 user_id: {user_id}")
    print(f"当前 thread_id: {thread_id}")
    print("MCP 工具：list_documents / search_documents")
    print("命令：/remember key=value  /preferences  /new  /exit")

    while True:
        try:
            text = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not text:
            continue
        if text in {"/exit", "/quit"}:
            print("再见。")
            break
        if text == "/new":
            thread_id = uuid.uuid4().hex[:8]
            print(f"已新建会话，thread_id: {thread_id}")
            continue
        if text.startswith("/remember "):
            payload = text[len("/remember ") :].strip()
            if "=" not in payload:
                print("用法：/remember key=value")
                continue
            key, value = payload.split("=", 1)
            try:
                preferences = save_preference(
                    store, user_id, key.strip(), value.strip()
                )
            except ValueError as exc:
                print(f"保存失败：{exc}")
                continue
            print(f"已记住：{key.strip()} = {value.strip()}")
            for item_key, item_value in sorted(preferences.items()):
                print(f"- {item_key}: {item_value}")
            # 偏好变化后重建 Agent，让新的 system prompt 生效。
            agent = build_mcp_agent(
                tools,
                checkpointer=checkpointer,
                store=store,
                user_id=user_id,
            )
            continue
        if text in {"/preferences", "/memory"}:
            preferences = load_preferences(store, user_id)
            if not preferences:
                print("当前用户还没有保存长期偏好。")
            else:
                print("当前用户长期偏好：")
                for key, value in sorted(preferences.items()):
                    print(f"- {key}: {value}")
            continue

        print("助手> ", end="", flush=True)
        await ask(text)

    return 0


async def run_mcp_mode(
    *,
    question: str | None,
    thread_id: str,
    user_id: str,
    store: Any,
    checkpoint_db: str,
    stream: bool = True,
) -> int:
    """CLI 的 MCP 模式入口：使用异步 SQLite checkpointer。"""
    from app.memory.checkpointer import open_async_checkpointer

    async with open_async_checkpointer(checkpoint_db) as checkpointer:
        return await run_mcp_cli(
            question=question,
            thread_id=thread_id,
            user_id=user_id,
            store=store,
            checkpointer=checkpointer,
            stream=stream,
        )


if __name__ == "__main__":
    from app.config import configure_logging
    from app.memory.checkpointer import open_async_checkpointer
    from app.memory.store import open_store

    settings = get_settings()
    configure_logging()

    async def _main() -> None:
        with open_store(settings.memory_db) as store:
            async with open_async_checkpointer(
                settings.checkpoint_db
            ) as checkpointer:
                await run_mcp_cli(
                    question=None,
                    thread_id="mcp-cli",
                    user_id=settings.default_user,
                    store=store,
                    checkpointer=checkpointer,
                )

    asyncio.run(_main())
