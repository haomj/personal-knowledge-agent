"""HITL Graph：生成报告 -> interrupt 审批 -> 保存/取消。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.hitl.report import draft_report, save_research_report
from app.state import AgentState
from app.text import clean_text, content_text

logger = logging.getLogger(__name__)

DraftFn = Callable[[str], tuple[str, str]]


class HITLState(AgentState):
    report_title: str
    report_content: str
    approved: bool
    feedback: str
    report_path: str
    status: str


def _last_human_text(state: HITLState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return clean_text(content_text(message.content))
    return ""


def initial_hitl_state(
    question: str, user_id: str = "default"
) -> dict[str, Any]:
    cleaned = clean_text(question)
    return {
        "messages": [HumanMessage(content=cleaned)],
        "user_id": user_id,
        "report_title": "",
        "report_content": "",
        "approved": False,
        "feedback": "",
        "report_path": "",
        "status": "",
    }


def build_hitl_graph(
    *,
    draft_fn: DraftFn | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    reports_dir: str | Path | None = None,
):
    """构建 HITL 图。

    draft_fn 可注入，便于离线测试；默认使用本地 RAG + LLM。
    """
    draft = draft_fn or draft_report

    def draft_report_node(state: HITLState) -> dict[str, Any]:
        question = _last_human_text(state)
        title, content = draft(question)
        logger.info("hitl.node.draft title=%r", title)
        return {
            "report_title": title,
            "report_content": content,
            "status": "drafted",
        }

    def approval_node(state: HITLState) -> dict[str, Any]:
        # 注意：恢复时本节点会从头重新执行；interrupt() 之前不能有副作用。
        payload = {
            "type": "approval_request",
            "action": "save_research_report",
            "title": state.get("report_title", ""),
            "content_preview": state.get("report_content", "")[:1200],
            "content_length": len(state.get("report_content", "")),
        }
        decision = interrupt(payload)

        if isinstance(decision, dict):
            approved = bool(decision.get("approved"))
            feedback = str(decision.get("feedback", ""))
        else:
            approved = bool(decision)
            feedback = ""

        logger.info(
            "hitl.node.approval approved=%s feedback=%r",
            approved,
            feedback,
        )
        return {
            "approved": approved,
            "feedback": feedback,
            "status": "approved" if approved else "rejected",
        }

    def save_report_node(state: HITLState) -> dict[str, Any]:
        try:
            path = save_research_report(
                state.get("report_title", "研究报告"),
                state.get("report_content", ""),
                reports_dir,
            )
        except Exception as exc:
            logger.exception("hitl.node.save failed")
            return {
                "status": "error",
                "feedback": f"保存失败：{exc}",
                "report_path": "",
            }
        return {"status": "saved", "report_path": str(path)}

    def cancel_node(state: HITLState) -> dict[str, Any]:
        logger.info("hitl.node.cancel")
        return {"status": "cancelled", "report_path": ""}

    def route_after_approval(state: HITLState) -> str:
        return "save_report" if state.get("approved") else "cancel"

    builder = StateGraph(HITLState)
    builder.add_node("draft_report", draft_report_node)
    builder.add_node("approval", approval_node)
    builder.add_node("save_report", save_report_node)
    builder.add_node("cancel", cancel_node)

    builder.add_edge(START, "draft_report")
    builder.add_edge("draft_report", "approval")
    builder.add_conditional_edges(
        "approval",
        route_after_approval,
        {"save_report": "save_report", "cancel": "cancel"},
    )
    builder.add_edge("save_report", END)
    builder.add_edge("cancel", END)

    compiled = builder.compile(checkpointer=checkpointer)
    logger.info(
        "hitl.graph.compiled checkpointer=%s reports_dir=%s",
        type(checkpointer).__name__ if checkpointer else "none",
        reports_dir,
    )
    return compiled
