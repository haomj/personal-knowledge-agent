"""真实模型冒烟测试：默认跳过，设置 RUN_LIVE_TESTS=1 后运行。"""

from __future__ import annotations

import os

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import build_graph

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1",
    reason="需要真实 API；设置 RUN_LIVE_TESTS=1 运行",
)


def test_deepseek_chat_and_multi_turn():
    graph = build_graph(checkpointer=InMemorySaver(), system_prompt="回答尽量简短。")
    config = {"configurable": {"thread_id": "live-multiturn"}}

    first = graph.invoke(
        {"messages": [HumanMessage(content="我叫 Adrian，请只回复：记住你了")]},
        config,
    )
    assert first["messages"][-1].content

    second = graph.invoke(
        {"messages": [HumanMessage(content="我叫什么名字？只回答名字。")]},
        config,
    )
    answer = second["messages"][-1].content
    assert "Adrian" in answer


def test_deepseek_streaming():
    graph = build_graph(system_prompt="回答尽量简短。")
    pieces: list[str] = []

    for chunk, metadata in graph.stream(
        {"messages": [HumanMessage(content="数到三")]},
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") == "agent" and isinstance(chunk.content, str):
            pieces.append(chunk.content)

    assert "".join(pieces).strip()
