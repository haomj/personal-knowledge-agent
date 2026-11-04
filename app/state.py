"""Graph 的共享状态定义。"""

from typing import Any

from langchain_core.documents import Document
from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Agent 的共享状态。

    Day 1 只使用 messages；
    Day 2 增加 RAG 检索过程需要的字段。
    """

    query: str
    documents: list[Document]
    sources: list[dict[str, Any]]
    retrieval_attempts: int
    retrieval_sufficient: bool
    grade_reason: str