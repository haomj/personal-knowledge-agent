"""Agentic RAG Graph。

结构：

    START -> retrieve -> grade
                          ├─ 足够 -> generate -> END
                          └─ 不足 -> rewrite -> retrieve
                                      （最多重试 max_retries 次）

生成节点只根据检索资料回答，并在 State 中保留来源，供 CLI/Web UI 展示。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore

from app.config import get_model, get_settings
from app.memory.store import preference_prompt
from app.rag.evaluator import LLMGrader, RelevanceGrade
from app.rag.retriever import (
    build_default_retriever,
    sources_from_documents,
)
from app.rag.rewriter import LLMQueryRewriter
from app.state import AgentState
from app.text import clean_content, clean_text, content_text

logger = logging.getLogger(__name__)


class RetrieverLike(Protocol):
    def retrieve(self, query: str) -> list[Document]: ...


class GraderLike(Protocol):
    def grade(
        self, question: str, documents: list[Document]
    ) -> RelevanceGrade: ...


class RewriterLike(Protocol):
    def rewrite(
        self, question: str, current_query: str | None = None
    ) -> str: ...


def _last_human_text(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return clean_text(content_text(message.content))
    return ""


def _clean_messages(messages):
    """调用模型前清洗消息，避免孤立代理项进入 JSON 请求。"""
    cleaned = []
    for message in messages:
        content = clean_content(message.content)
        if content != message.content:
            message = message.model_copy(update={"content": content})
        cleaned.append(message)
    return cleaned


def _format_context(documents, limit: int = 4) -> str:
    if not documents:
        return "（没有检索到相关资料）"
    parts: list[str] = []
    for index, document in enumerate(documents[:limit], 1):
        source = document.metadata.get("source", "unknown")
        title = document.metadata.get("title", "")
        content = document.page_content.strip()
        if len(content) > 1200:
            content = content[:1200] + "..."
        parts.append(f"[{index}] 来源: {source}\n标题: {title}\n{content}")
    return "\n\n".join(parts)


def initial_rag_state(
    question: str, user_id: str = "default"
) -> dict[str, Any]:
    """构造一次 RAG 调用的初始 State。"""
    cleaned = clean_text(question)
    return {
        "messages": [HumanMessage(content=cleaned)],
        "query": cleaned,
        "documents": [],
        "sources": [],
        "retrieval_attempts": 0,
        "retrieval_sufficient": False,
        "grade_reason": "",
        "user_id": clean_text(user_id) or "default",
    }


def build_rag_graph(
    model: BaseChatModel | None = None,
    *,
    retriever: RetrieverLike | None = None,
    grader: GraderLike | None = None,
    rewriter: RewriterLike | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    system_prompt: str | None = None,
    top_k: int | None = None,
    max_retries: int | None = None,
    store: BaseStore | None = None,
) -> Any:
    """构建 Agentic RAG 图。

    参数可以注入，便于单元测试：
    - retriever：默认使用 Chroma；
    - grader / rewriter：默认使用 DeepSeek 的非流式模型；
    - model：生成节点使用的模型，默认开启流式。
    """
    settings = get_settings()
    chat_model = model or get_model()
    knowledge = retriever or build_default_retriever(top_k=top_k)
    grade_model = grader or LLMGrader(get_model(streaming=False))
    rewrite_model = rewriter or LLMQueryRewriter(get_model(streaming=False))
    prompt = clean_text(system_prompt or settings.system_prompt)
    retry_limit = settings.rag_max_retries if max_retries is None else max_retries

    def retrieve_node(state: AgentState) -> dict[str, Any]:
        question = _last_human_text(state)
        query = clean_text(state.get("query") or question)
        attempts = int(state.get("retrieval_attempts", 0)) + 1
        documents = knowledge.retrieve(query)
        sources = sources_from_documents(documents)
        logger.info(
            "rag.node.retrieve attempt=%d query=%r docs=%d",
            attempts,
            query,
            len(documents),
        )
        return {
            "query": query,
            "documents": documents,
            "sources": sources,
            "retrieval_attempts": attempts,
        }

    def grade_node(state: AgentState) -> dict[str, Any]:
        question = _last_human_text(state)
        documents = state.get("documents", [])
        grade = grade_model.grade(question, documents)
        logger.info(
            "rag.node.grade sufficient=%s reason=%s",
            grade.sufficient,
            grade.reason,
        )
        return {
            "retrieval_sufficient": grade.sufficient,
            "grade_reason": grade.reason,
        }

    def rewrite_node(state: AgentState) -> dict[str, Any]:
        question = _last_human_text(state)
        current_query = state.get("query") or question
        new_query = rewrite_model.rewrite(question, current_query)
        logger.info("rag.node.rewrite query=%r", new_query)
        return {"query": new_query}

    def generate_node(state: AgentState) -> dict[str, list[AIMessage]]:
        documents = state.get("documents", [])
        sufficient = bool(state.get("retrieval_sufficient"))
        context = _format_context(documents)

        if sufficient:
            instruction = (
                "检索资料足以回答用户问题。请只根据下面的资料回答，"
                "并在关键结论后用 [来源: 文件路径] 标注出处。"
                "引用格式必须写成 [来源: data/docs/xxx.md]，不要只写 [1] 这样的编号。"
                "如果资料中有冲突，请指出冲突。"
            )
        else:
            instruction = (
                "检索资料不足或未通过相关性检查。请明确说明知识库中"
                "没有足够信息来确认答案，不要编造事实或来源；"
                "如果资料中有一部分相关内容，可以说明它覆盖了什么、缺少什么。"
                "如果引用了部分资料，同样使用 [来源: 文件路径] 标注。"
            )

        system_content = (
            f"{prompt}\n\n{instruction}\n\n检索资料：\n{context}"
        )
        user_id = state.get("user_id", "default")
        preference_block = preference_prompt(store, user_id)
        if preference_block:
            system_content = f"{system_content}\n\n{preference_block}"
        request = [
            SystemMessage(content=system_content),
            *_clean_messages(state["messages"]),
        ]

        started = time.perf_counter()
        logger.info(
            "rag.node.generate sufficient=%s docs=%d user_id=%s memory=%s",
            sufficient,
            len(documents),
            user_id,
            bool(preference_block),
        )
        response = chat_model.invoke(request)
        elapsed = time.perf_counter() - started
        usage = response.usage_metadata or {}
        logger.info(
            "rag.node.generate.done elapsed=%.2fs input_tokens=%s output_tokens=%s",
            elapsed,
            usage.get("input_tokens"),
            usage.get("output_tokens"),
        )
        return {"messages": [response]}

    def route_after_grade(state: AgentState) -> str:
        if state.get("retrieval_sufficient"):
            return "generate"
        if int(state.get("retrieval_attempts", 0)) <= retry_limit:
            return "rewrite"
        return "generate"

    builder = StateGraph(AgentState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade", grade_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("generate", END)

    compiled = builder.compile(checkpointer=checkpointer)
    logger.info(
        "rag.graph.compiled model=%s top_k=%s max_retries=%s store=%s",
        getattr(chat_model, "model_name", type(chat_model).__name__),
        getattr(knowledge, "k", "injected"),
        retry_limit,
        type(store).__name__ if store else "none",
    )
    return compiled
