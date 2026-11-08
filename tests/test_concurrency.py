"""Day 10：并发请求与 thread 隔离测试。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import build_graph
from app.web.server import _thread_key


class EchoChatModel(BaseChatModel):
    """把最后一条 HumanMessage 原样回显，便于检测线程串线。"""

    @property
    def _llm_type(self) -> str:
        return "echo-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        last_human = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, HumanMessage)
            ),
            None,
        )
        text = str(last_human.content) if last_human else ""
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=f"echo:{text}")
                )
            ]
        )


def test_concurrent_threads_do_not_mix():
    graph = build_graph(
        model=EchoChatModel(),
        checkpointer=InMemorySaver(),
        system_prompt="并发测试",
    )

    def ask(thread_id: str, text: str):
        return graph.invoke(
            {
                "messages": [HumanMessage(content=text)],
                "user_id": "u1",
            },
            {"configurable": {"thread_id": thread_id}},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(ask, "thread-a", "alpha")
        second = executor.submit(ask, "thread-b", "beta")
        result_a = first.result()
        result_b = second.result()

    assert result_a["messages"][-1].content == "echo:alpha"
    assert result_b["messages"][-1].content == "echo:beta"

    follow_a = ask("thread-a", "alpha-2")
    follow_b = ask("thread-b", "beta-2")

    assert [m.content for m in follow_a["messages"]] == [
        "alpha",
        "echo:alpha",
        "alpha-2",
        "echo:alpha-2",
    ]
    assert [m.content for m in follow_b["messages"]] == [
        "beta",
        "echo:beta",
        "beta-2",
        "echo:beta-2",
    ]


def test_thread_key_includes_mode_and_user():
    assert _thread_key("rag", "u1", "t1") != _thread_key(
        "rag", "u2", "t1"
    )
    assert _thread_key("rag", "u1", "t1") != _thread_key(
        "hitl", "u1", "t1"
    )
    assert _thread_key("rag", "u1", "t1") == _thread_key(
        "rag", "u1", "t1"
    )
