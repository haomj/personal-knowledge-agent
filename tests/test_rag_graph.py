"""RAG Graph 的路由测试：用假检索器和假评分器，不访问网络。"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from app.rag.evaluator import RelevanceGrade
from app.rag.graph import build_rag_graph, initial_rag_state


class StaticChatModel(BaseChatModel):
    answer: str
    calls: list[list[BaseMessage]] = Field(default_factory=list)

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
        self.calls.append(list(messages))
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.answer))]
        )


class FakeRetriever:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.queries: list[str] = []

    def retrieve(self, query: str) -> list[Document]:
        self.queries.append(query)
        return self.documents


class StaticGrader:
    def __init__(self, grades: list[RelevanceGrade]):
        self.grades = list(grades)
        self.calls = 0

    def grade(self, question: str, documents: list[Document]) -> RelevanceGrade:
        self.calls += 1
        return self.grades.pop(0)


class StaticRewriter:
    def __init__(self, query: str):
        self.query = query
        self.calls = 0

    def rewrite(self, question: str, current_query: str | None = None) -> str:
        self.calls += 1
        return self.query


def _documents() -> list[Document]:
    return [
        Document(
            page_content="Agentic RAG 最多改写查询一次，然后再次检索。",
            metadata={
                "source": "data/docs/03-agentic-rag.md",
                "title": "Agentic RAG",
                "score": 0.9,
            },
        )
    ]


def _build_graph(grader, rewriter=None, retriever=None, answer="最终答案"):
    return build_rag_graph(
        model=StaticChatModel(answer=answer),
        retriever=retriever or FakeRetriever(_documents()),
        grader=grader,
        rewriter=rewriter or StaticRewriter("改写后的查询"),
        system_prompt="测试提示词",
        max_retries=1,
    )


def test_sufficient_first_retrieval_generates_without_rewrite():
    retriever = FakeRetriever(_documents())
    rewriter = StaticRewriter("不应被调用")
    graph = _build_graph(
        StaticGrader([RelevanceGrade(sufficient=True, reason="足够")]),
        rewriter=rewriter,
        retriever=retriever,
        answer="根据资料回答",
    )

    result = graph.invoke(initial_rag_state("Agentic RAG 重试几次？"))

    assert result["messages"][-1].content == "根据资料回答"
    assert result["retrieval_attempts"] == 1
    assert result["retrieval_sufficient"] is True
    assert rewriter.calls == 0
    assert retriever.queries == ["Agentic RAG 重试几次？"]
    assert result["sources"][0]["source"] == "data/docs/03-agentic-rag.md"


def test_insufficient_then_sufficient_rewrites_and_retrieves_again():
    retriever = FakeRetriever(_documents())
    rewriter = StaticRewriter("Agentic RAG 查询改写 重试上限")
    graph = _build_graph(
        StaticGrader(
            [
                RelevanceGrade(sufficient=False, reason="缺少重试次数"),
                RelevanceGrade(sufficient=True, reason="第二次足够"),
            ]
        ),
        rewriter=rewriter,
        retriever=retriever,
    )

    result = graph.invoke(initial_rag_state("那个 RAG 的重试怎么算？"))

    assert retriever.queries == [
        "那个 RAG 的重试怎么算？",
        "Agentic RAG 查询改写 重试上限",
    ]
    assert rewriter.calls == 1
    assert result["retrieval_attempts"] == 2
    assert result["retrieval_sufficient"] is True


def test_insufficient_twice_stops_after_one_rewrite():
    retriever = FakeRetriever(_documents())
    rewriter = StaticRewriter("第二次查询")
    graph = _build_graph(
        StaticGrader(
            [
                RelevanceGrade(sufficient=False, reason="不足"),
                RelevanceGrade(sufficient=False, reason="仍然不足"),
            ]
        ),
        rewriter=rewriter,
        retriever=retriever,
        answer="知识库信息不足，无法确认。",
    )

    result = graph.invoke(initial_rag_state("知识库没有的问题"))

    assert retriever.queries == ["知识库没有的问题", "第二次查询"]
    assert rewriter.calls == 1
    assert result["retrieval_attempts"] == 2
    assert result["retrieval_sufficient"] is False
    assert result["messages"][-1].content == "知识库信息不足，无法确认。"
