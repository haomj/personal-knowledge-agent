"""Query Rewriting：检索结果不足时改写查询。"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from app.text import clean_text, content_text

logger = logging.getLogger(__name__)


class LLMQueryRewriter:
    """把用户问题改写成更适合向量检索的查询。"""

    def __init__(self, model: BaseChatModel):
        self.model = model

    def rewrite(self, question: str, current_query: str | None = None) -> str:
        fallback = current_query or question
        prompt = (
            "你是技术文档检索查询改写器。\n"
            "把用户问题改写成一句简短、具体的检索查询。\n"
            "要求：保留专有名词、API 名称和数字；补全代词指代；"
            "不要回答问题；不要加引号或解释。\n\n"
            f"用户问题：{question}\n"
            f"当前查询：{fallback}\n\n"
            "只输出改写后的查询："
        )
        try:
            response = self.model.invoke(prompt)
            rewritten = content_text(response.content).strip().strip("\"'`")
            rewritten = clean_text(rewritten)
            if not rewritten:
                return fallback
            logger.info(
                "rag.rewrite from=%r to=%r", fallback, rewritten
            )
            return rewritten
        except Exception as exc:
            logger.warning("rag.rewrite failed: %s", exc)
            return fallback
