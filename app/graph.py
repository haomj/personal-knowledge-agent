"""Day 1：最小可运行 Agent Graph。

图结构：

    START -> agent -> END

agent 节点读取 State 中的消息历史，调用模型，把 AIMessage 写回 State。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.config import get_model, get_settings
from app.state import AgentState
from app.text import clean_content, clean_text

logger = logging.getLogger(__name__)


def _clean_messages(messages):
    """在调用模型前清洗消息内容，防止孤立代理项进入 JSON 请求。"""
    cleaned = []
    changed = False
    for message in messages:
        content = clean_content(message.content)
        if content != message.content:
            message = message.model_copy(update={"content": content})
            changed = True
        cleaned.append(message)
    if changed:
        logger.warning("State 中存在无法编码的字符，已在调用模型前替换为 U+FFFD")
    return cleaned


def build_graph(
    model: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    *,
    system_prompt: str | None = None,
) -> Any:
    """构建并编译 Day 1 的 Agent 图。

    Args:
        model: 便于测试注入的模型；默认从环境变量创建 deepseek-flash。
        checkpointer: 开启后可用 thread_id 保存/恢复会话（Day 3 会换成 SQLite）。
        system_prompt: 系统提示词；默认读取配置。
    """
    model = model or get_model()
    prompt = clean_text(system_prompt or get_settings().system_prompt)

    def agent_node(state: AgentState) -> dict[str, list[AIMessage]]:
        # 节点只返回“增量” {"messages": [...]}，由 add_messages reducer 合并。
        messages = state["messages"]
        # 系统提示词不写入 State，避免每轮都把它累积进历史；调用时临时拼在消息列表前面。
        request = [SystemMessage(content=prompt), *_clean_messages(messages)]

        started = time.perf_counter()
        logger.info("agent.start messages=%d", len(messages))
        response = model.invoke(request)
        elapsed = time.perf_counter() - started

        usage = response.usage_metadata or {}
        logger.info(
            "agent.end elapsed=%.2fs input_tokens=%s output_tokens=%s",
            elapsed,
            usage.get("input_tokens"),
            usage.get("output_tokens"),
        )
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)

    compiled = builder.compile(checkpointer=checkpointer)
    logger.info(
        "graph.compiled model=%s checkpointer=%s",
        getattr(model, "model_name", type(model).__name__),
        type(checkpointer).__name__ if checkpointer else "none",
    )
    return compiled


def build_chat_graph() -> Any:
    """CLI 使用的图：带内存 checkpointer，同一 thread 内保留多轮历史。"""
    return build_graph(checkpointer=InMemorySaver())
