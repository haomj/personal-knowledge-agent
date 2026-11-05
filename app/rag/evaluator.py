"""Retrieval Grading：判断检索结果是否足以回答问题。"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RelevanceGrade(BaseModel):
    """检索相关性评分结果。"""

    sufficient: bool = Field(
        description="检索资料是否足以回答用户问题"
    )
    reason: str = Field(description="判断理由，简短说明缺少什么或为什么足够")


class LLMGrader:
    """用 LLM 做语义层面的检索评分。

    DeepSeek 的默认 JSON schema structured output 不可用，
    因此显式使用 function calling 方式。
    """

    def __init__(self, model: BaseChatModel):
        self.model = model
        self.structured = model.with_structured_output(
            RelevanceGrade, method="function_calling"
        )

    def grade(self, question: str, documents: list[Document]) -> RelevanceGrade:
        if not documents:
            return RelevanceGrade(
                sufficient=False, reason="没有检索到任何文档"
            )

        context = _format_documents(documents)
        prompt = (
            "你是检索质量评估器。判断给定资料是否足以回答用户问题。\n"
            "只根据资料内容判断，不要用你自己的背景知识补全。\n"
            "如果资料覆盖了问题的关键事实，sufficient=true；\n"
            "如果资料缺少关键信息、只沾边或互相矛盾，sufficient=false。\n\n"
            f"用户问题：\n{question}\n\n"
            f"检索资料：\n{context}"
        )

        try:
            result = self.structured.invoke(prompt)
            if isinstance(result, RelevanceGrade):
                return result
            if isinstance(result, BaseModel):
                result = result.model_dump()
            return RelevanceGrade.model_validate(result)
        except Exception as exc:
            logger.warning("rag.grade failed, fallback to heuristic: %s", exc)
            # 降级策略：有检索结果且最高分不算太低，就交给生成节点；
            # 否则让流程进入查询改写。
            top_score = max(
                float(document.metadata.get("score", 0.0))
                for document in documents
            )
            sufficient = top_score >= 0.2
            return RelevanceGrade(
                sufficient=sufficient,
                reason=f"评分调用失败，按最高分 {top_score:.3f} 降级判断",
            )


def _format_documents(documents: list[Document], limit: int = 4) -> str:
    parts: list[str] = []
    for index, document in enumerate(documents[:limit], 1):
        source = document.metadata.get("source", "unknown")
        content = document.page_content.strip()
        if len(content) > 800:
            content = content[:800] + "..."
        parts.append(f"[{index}] 来源: {source}\n{content}")
    return "\n\n".join(parts)
