"""Middleware 演示 Agent：把日志、权限、重试串成一个工具调用 Agent。"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.config import get_model
from app.middleware.logging import LoggingMiddleware
from app.middleware.permission import PermissionMiddleware
from app.middleware.retry import RetryMiddleware
from app.middleware.tools import (
    UserContext,
    lookup_with_retry,
    read_knowledge,
    save_research_report,
)
from app.text import content_text

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是 Middleware 演示助手。可用工具：
- read_knowledge(query)：只读检索本地知识库；
- lookup_with_retry(query)：模拟不稳定的只读查询，由重试中间件处理；
- save_research_report(title, content)：保存正式报告，高风险，仅 admin 可调用。

用户要求调用工具时必须真实调用，不要假装调用。
保存报告时权限中间件会检查当前用户角色；不要绕过权限。
"""


def build_middleware_agent(*, checkpointer: Any = None):
    return create_agent(
        get_model(streaming=False),
        tools=[read_knowledge, save_research_report, lookup_with_retry],
        middleware=[
            LoggingMiddleware(),
            PermissionMiddleware(),
            RetryMiddleware(),
        ],
        context_schema=UserContext,
        checkpointer=checkpointer,
        system_prompt=SYSTEM_PROMPT,
    )


def _print_trace(messages: list[Any]) -> None:
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                print(
                    f"[Middleware] 模型请求工具 {call.get('name')} "
                    f"{call.get('args', {})}"
                )
        elif isinstance(message, ToolMessage):
            text = content_text(message.content)
            preview = text[:300] + ("..." if len(text) > 300 else "")
            print(
                f"[Middleware] 工具 {message.name or 'tool'} "
                f"status={message.status}: {preview}"
            )


def _ask(
    agent: Any,
    text: str,
    thread_id: str,
    context: UserContext,
) -> str:
    result = agent.invoke(
        {"messages": [HumanMessage(content=text)]},
        config={"configurable": {"thread_id": thread_id}},
        context=context,
    )
    messages = result["messages"]
    _print_trace(messages)
    answer = content_text(messages[-1].content)
    print(answer)
    return answer


def run_middleware_cli(
    agent: Any,
    *,
    question: str | None,
    thread_id: str,
    user_id: str,
    role: str,
) -> int:
    context = UserContext(user_id=user_id, role=role)
    print("个人知识研究助手 v0.7 Middleware")
    print(f"当前 user_id: {user_id} / role: {role}")
    print(f"当前 thread_id: {thread_id}")
    print("工具：read_knowledge / lookup_with_retry / save_research_report")
    print("命令：/role admin|viewer  /new  /exit")

    if question:
        _ask(agent, question, thread_id, context)
        return 0

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
        if text.startswith("/role"):
            parts = text.split()
            if len(parts) == 2 and parts[1] in {"admin", "viewer"}:
                role = parts[1]
                context = UserContext(user_id=user_id, role=role)
                print(f"角色已切换为：{role}")
            else:
                print("用法：/role admin 或 /role viewer")
            continue

        _ask(agent, text, thread_id, context)

    return 0
