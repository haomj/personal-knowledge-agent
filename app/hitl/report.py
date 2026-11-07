"""HITL 报告：生成草稿、幂等保存到 data/reports/。"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from app.config import get_model, get_settings
from app.rag.retriever import build_default_retriever
from app.text import content_text

logger = logging.getLogger(__name__)


def _format_docs(documents, limit: int = 4) -> str:
    if not documents:
        return "（本地知识库没有检索到相关资料）"
    parts: list[str] = []
    for index, document in enumerate(documents[:limit], 1):
        source = document.metadata.get("source", "unknown")
        content = document.page_content.strip()
        if len(content) > 1200:
            content = content[:1200] + "..."
        parts.append(f"[{index}] 来源: {source}\n{content}")
    return "\n\n".join(parts)


def _parse_report(text: str, question: str) -> tuple[str, str]:
    """把模型输出拆成标题和正文；模型没有按格式输出时做降级。"""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip() or question[:60]
        content = "\n".join(lines[1:]).strip() or text
        return title, content
    title = question.strip()[:60] or "研究报告"
    return title, text.strip()


def draft_report(question: str) -> tuple[str, str]:
    """默认草稿生成：本地 RAG 检索 + 非流式 LLM。"""
    try:
        documents = build_default_retriever().retrieve(question)
    except Exception as exc:
        logger.warning("hitl.retrieve failed: %s", exc)
        documents = []

    prompt = (
        "你是研究助手。请生成一份简洁的中文研究报告。\n"
        "第一行必须是标题，以 '# ' 开头；空一行后是正文。\n"
        "正文只基于下面的本地资料，关键结论使用 [来源: 文件路径] 标注；"
        "资料不足时明确说明，不要编造。\n\n"
        f"研究问题：{question}\n\n"
        f"本地资料：\n{_format_docs(documents)}"
    )
    response = get_model(streaming=False).invoke(prompt)
    text = content_text(response.content)
    title, content = _parse_report(text, question)
    logger.info(
        "hitl.draft title=%r content_chars=%d docs=%d",
        title,
        len(content),
        len(documents),
    )
    return title, content


def _slugify(title: str) -> str:
    slug = re.sub(r"[\\/:*?\"<>|\s]+", "-", title).strip("-")
    return (slug or "report")[:60]


def save_research_report(
    title: str,
    content: str,
    reports_dir: str | Path | None = None,
) -> Path:
    """保存报告，使用标题和内容哈希生成稳定文件名。

    同样的 title + content 重复保存会返回同一个路径，不产生重复文件。
    """
    directory = Path(reports_dir or get_settings().reports_dir)
    directory.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256(
        f"{title}\n{content}".encode("utf-8")
    ).hexdigest()[:8]
    path = directory / f"{_slugify(title)}-{digest}.md"
    rendered = f"# {title}\n\n{content}\n"

    if path.exists():
        logger.info("hitl.save idempotent path=%s", path)
        return path

    path.write_text(rendered, encoding="utf-8")
    logger.info("hitl.save path=%s chars=%d", path, len(content))
    return path
