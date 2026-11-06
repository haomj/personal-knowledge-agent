"""MultiAgent 真实端到端测试：默认跳过。"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.multiagent.supervisor import build_multiagent
from app.text import content_text

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1"
    or not os.getenv("TAVILY_API_KEY")
    or not Path("data/chroma").exists(),
    reason="需要 Tavily、DeepSeek、Embedding API 和已构建的 Chroma 索引",
)


def test_supervisor_calls_both_agents_and_aggregates():
    async def _run():
        agent = build_multiagent(
            checkpointer=InMemorySaver(),
            user_id="multi-live",
            system_prompt="回答尽量简短。",
        )
        return await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "结合本地知识库中 Agentic RAG 的重试策略，"
                            "以及 2026 年 LangGraph 的最新版本信息，"
                            "给我一个简短总结。"
                        )
                    )
                ]
            },
            config={"configurable": {"thread_id": "multi-live"}},
        )

    result = asyncio.run(_run())
    tool_names: list[str] = []
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            tool_names.extend(
                call["name"] for call in message.tool_calls
            )

    assert "ask_knowledge_agent" in tool_names
    assert "ask_research_agent" in tool_names

    answer = content_text(result["messages"][-1].content)
    assert "Agentic RAG" in answer
    assert "http" in answer
