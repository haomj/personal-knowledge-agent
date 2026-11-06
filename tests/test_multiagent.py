"""MultiAgent 离线测试：预算、Knowledge Agent、Research Agent。"""

from __future__ import annotations

import asyncio

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.multiagent.knowledge import run_knowledge_agent
from app.multiagent.research import run_research_agent
from app.multiagent.supervisor import TaskBudget


class StaticChatModel(BaseChatModel):
    answer: str

    @property
    def _llm_type(self) -> str:
        return "static-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.answer))]
        )


async def fake_knowledge_search(query: str, k: int):
    return [
        {
            "source": "data/docs/03-agentic-rag.md",
            "title": "Agentic RAG",
            "score": 0.9,
            "content": "Agentic RAG 最多改写查询一次，然后再次检索。",
        }
    ]


async def fake_research_search(query: str, max_results: int):
    return [
        {
            "title": "LangGraph releases",
            "url": "https://github.com/langchain-ai/langgraph/releases",
            "content": "LangGraph latest release information.",
        }
    ]


def test_task_budget_limits():
    async def _run():
        budget = TaskBudget(max_tasks=3, max_per_agent=2)
        results = [
            await budget.acquire("knowledge"),
            await budget.acquire("knowledge"),
            await budget.acquire("knowledge"),
            await budget.acquire("research"),
        ]
        return results

    assert asyncio.run(_run()) == [True, True, False, True]


def test_knowledge_agent_returns_answer_and_sources():
    result = asyncio.run(
        run_knowledge_agent(
            "Agentic RAG 最多重试几次？",
            search=fake_knowledge_search,
            model=StaticChatModel(answer="最多 1 次改写、2 次检索"),
        )
    )
    assert result.agent == "knowledge"
    assert "1 次改写" in result.answer
    assert result.sources == ["data/docs/03-agentic-rag.md"]


def test_research_agent_returns_answer_and_urls():
    result = asyncio.run(
        run_research_agent(
            "LangGraph latest release",
            search=fake_research_search,
            model=StaticChatModel(answer="最新版本信息见 release 页面"),
        )
    )
    assert result.agent == "research"
    assert "release" in result.answer
    assert result.sources == [
        "https://github.com/langchain-ai/langgraph/releases"
    ]
