"""Day 3 记忆测试：SQLite 重启恢复、跨 thread 偏好共享、用户隔离。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import Field

from app.graph import build_graph
from app.memory.checkpointer import open_checkpointer
from app.memory.store import (
    load_preferences,
    open_store,
    save_preference,
)
from app.rag.evaluator import RelevanceGrade
from app.rag.graph import build_rag_graph, initial_rag_state


class RecordingChatModel(BaseChatModel):
    responses: list[AIMessage]
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "recording-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        self.calls.append(list(messages))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return ChatResult(generations=[ChatGeneration(message=self.responses[index])])


class FakeRetriever:
    def retrieve(self, query: str) -> list[Document]:
        return [
            Document(
                page_content="测试资料",
                metadata={
                    "source": "data/docs/test.md",
                    "title": "测试",
                    "score": 0.9,
                },
            )
        ]


class FakeGrader:
    def grade(self, question: str, documents: list[Document]) -> RelevanceGrade:
        return RelevanceGrade(sufficient=True, reason="测试足够")


class FakeRewriter:
    def rewrite(self, question: str, current_query: str | None = None) -> str:
        return current_query or question


def test_sqlite_checkpointer_restores_thread_after_reopen(tmp_path: Path):
    db_path = tmp_path / "checkpoints.sqlite"
    config = {"configurable": {"thread_id": "thread-restart"}}

    first_model = RecordingChatModel(
        responses=[AIMessage(content="第一次回答")]
    )
    with open_checkpointer(db_path) as checkpointer:
        graph = build_graph(
            model=first_model,
            checkpointer=checkpointer,
            system_prompt="测试",
        )
        graph.invoke(
            {
                "messages": [HumanMessage(content="我叫 Adrian")],
                "user_id": "u1",
            },
            config,
        )

    second_model = RecordingChatModel(
        responses=[AIMessage(content="重启后回答")]
    )
    with open_checkpointer(db_path) as checkpointer:
        graph = build_graph(
            model=second_model,
            checkpointer=checkpointer,
            system_prompt="测试",
        )
        result = graph.invoke(
            {
                "messages": [HumanMessage(content="我叫什么名字？")],
                "user_id": "u1",
            },
            config,
        )

    seen = [message.content for message in second_model.calls[0]]
    assert "我叫 Adrian" in seen
    assert "第一次回答" in seen
    assert len(result["messages"]) == 4


def test_sqlite_store_persists_preferences_after_reopen(tmp_path: Path):
    db_path = tmp_path / "memory.sqlite"

    with open_store(db_path) as store:
        save_preference(store, "u1", "language", "中文")
        save_preference(store, "u1", "style", "简洁")

    with open_store(db_path) as store:
        preferences = load_preferences(store, "u1")

    assert preferences == {"language": "中文", "style": "简洁"}


def test_preferences_shared_across_threads_but_isolated_by_user():
    store = InMemoryStore()
    save_preference(store, "u1", "language", "中文")
    model = RecordingChatModel(
        responses=[
            AIMessage(content="A"),
            AIMessage(content="B"),
            AIMessage(content="C"),
        ]
    )
    graph = build_graph(
        model=model,
        checkpointer=InMemorySaver(),
        store=store,
        system_prompt="测试",
    )

    graph.invoke(
        {"messages": [HumanMessage(content="会话 A")], "user_id": "u1"},
        {"configurable": {"thread_id": "thread-a"}},
    )
    graph.invoke(
        {"messages": [HumanMessage(content="会话 B")], "user_id": "u1"},
        {"configurable": {"thread_id": "thread-b"}},
    )
    graph.invoke(
        {"messages": [HumanMessage(content="会话 C")], "user_id": "u2"},
        {"configurable": {"thread_id": "thread-c"}},
    )

    assert "language: 中文" in model.calls[0][0].content
    assert "language: 中文" in model.calls[1][0].content
    assert "language: 中文" not in model.calls[2][0].content


def test_rag_generation_receives_user_preferences():
    store = InMemoryStore()
    save_preference(store, "u1", "language", "中文")
    model = RecordingChatModel(responses=[AIMessage(content="RAG 回答")])
    graph = build_rag_graph(
        model=model,
        retriever=FakeRetriever(),
        grader=FakeGrader(),
        rewriter=FakeRewriter(),
        store=store,
        system_prompt="测试",
        max_retries=1,
    )

    graph.invoke(initial_rag_state("测试问题", user_id="u1"))
    graph.invoke(initial_rag_state("测试问题", user_id="u2"))

    assert "language: 中文" in model.calls[0][0].content
    assert "language: 中文" not in model.calls[1][0].content
