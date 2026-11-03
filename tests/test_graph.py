"""Graph 的单元测试：不访问网络，用 RecordingChatModel 替身。"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from app.graph import build_graph


class RecordingChatModel(BaseChatModel):
    """测试替身：记录每次收到的消息，并按顺序返回预设回答。"""

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


def test_single_turn_builds_prompt_and_returns_answer():
    model = RecordingChatModel(responses=[AIMessage(content="你好，我是助手")])
    graph = build_graph(model=model, system_prompt="你是测试助手")

    result = graph.invoke({"messages": [HumanMessage(content="你好")]})

    assert result["messages"][-1].content == "你好，我是助手"
    assert isinstance(model.calls[0][0], SystemMessage)
    assert model.calls[0][0].content == "你是测试助手"
    assert model.calls[0][-1].content == "你好"


def test_same_thread_keeps_history():
    model = RecordingChatModel(
        responses=[AIMessage(content="记住了"), AIMessage(content="你叫 Adrian")]
    )
    graph = build_graph(model=model, checkpointer=InMemorySaver(), system_prompt="测试")
    config = {"configurable": {"thread_id": "t-1"}}

    graph.invoke({"messages": [HumanMessage(content="我叫 Adrian")]}, config)
    result = graph.invoke({"messages": [HumanMessage(content="我叫什么？")]}, config)

    # 第二次调用时，模型应看到：system + 第一轮 human/ai + 第二轮 human
    second_call = model.calls[1]
    assert [m.content for m in second_call] == [
        "测试",
        "我叫 Adrian",
        "记住了",
        "我叫什么？",
    ]
    assert len(result["messages"]) == 4


def test_different_threads_are_isolated():
    model = RecordingChatModel(
        responses=[AIMessage(content="A"), AIMessage(content="B")]
    )
    graph = build_graph(model=model, checkpointer=InMemorySaver(), system_prompt="测试")

    graph.invoke(
        {"messages": [HumanMessage(content="会话A")]},
        {"configurable": {"thread_id": "thread-a"}},
    )
    graph.invoke(
        {"messages": [HumanMessage(content="会话B")]},
        {"configurable": {"thread_id": "thread-b"}},
    )

    assert [m.content for m in model.calls[1]] == ["测试", "会话B"]


def test_stream_messages_mode_returns_answer():
    model = RecordingChatModel(responses=[AIMessage(content="流式回答")])
    graph = build_graph(model=model, system_prompt="测试")

    chunks = list(
        graph.stream(
            {"messages": [HumanMessage(content="你好")]},
            stream_mode="messages",
        )
    )
    text = "".join(
        chunk.content
        for chunk, metadata in chunks
        if metadata.get("langgraph_node") == "agent"
        and isinstance(chunk.content, str)
    )
    assert text == "流式回答"


def test_surrogate_in_message_is_cleaned_before_model_call():
    model = RecordingChatModel(responses=[AIMessage(content="ok")])
    graph = build_graph(model=model, system_prompt="测试")

    graph.invoke({"messages": [HumanMessage(content="你好\udcef，我是 Adrian")]})

    assert model.calls[0][-1].content == "你好�，我是 Adrian"
