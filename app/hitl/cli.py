"""HITL CLI：暂停、显示审批请求、通过/拒绝、重启恢复。"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.types import Command

from app.hitl.graph import initial_hitl_state

logger = logging.getLogger(__name__)


def _pending_interrupt(snapshot: Any) -> Any | None:
    if snapshot is None:
        return None
    for task in getattr(snapshot, "tasks", ()) or ():
        for item in getattr(task, "interrupts", ()) or ():
            return item
    return None


def _print_request(payload: dict[str, Any]) -> None:
    print("\n=== HITL 审批请求 ===")
    print(f"动作：{payload.get('action', 'save_research_report')}")
    print(f"标题：{payload.get('title', '')}")
    print(f"正文长度：{payload.get('content_length', 0)} 字符")
    print("预览：")
    print(payload.get("content_preview", "")[:1200])
    print("=====================")


def _make_decision(payload: dict[str, Any], mode: str) -> dict[str, Any] | None:
    if mode == "approve":
        return {"approved": True, "feedback": "CLI 自动批准"}
    if mode == "reject":
        return {"approved": False, "feedback": "CLI 自动拒绝"}
    if mode == "pause":
        return None

    try:
        answer = input(
            "是否保存报告？[y] 通过 / [n] 拒绝 / [p] 暂停退出："
        ).strip().lower()
    except EOFError:
        answer = "n"

    if answer in {"p", "pause", "暂停"}:
        return None
    approved = answer in {"y", "yes", "是", "通过", "保存"}
    return {
        "approved": approved,
        "feedback": "" if approved else answer or "用户拒绝",
    }


def run_hitl_question(
    graph: Any,
    *,
    question: str,
    thread_id: str,
    user_id: str,
    decision_mode: str = "ask",
) -> str:
    """执行一次 HITL 流程；返回 saved/rejected/error/paused。"""
    config = {"configurable": {"thread_id": thread_id}}

    # 如果同一 thread 之前停在审批节点，直接恢复，不重新生成草稿。
    snapshot = graph.get_state(config)
    pending = _pending_interrupt(snapshot)

    if pending is None:
        graph.invoke(
            initial_hitl_state(question, user_id=user_id),
            config,
        )
        snapshot = graph.get_state(config)
        pending = _pending_interrupt(snapshot)

    if pending is None:
        values = snapshot.values if snapshot else {}
        print(f"流程结束，状态：{values.get('status', 'unknown')}")
        return str(values.get("status", "unknown"))

    payload = pending.value if hasattr(pending, "value") else pending
    if not isinstance(payload, dict):
        payload = {"title": "", "content_preview": str(payload)}
    _print_request(payload)

    decision = _make_decision(payload, decision_mode)
    if decision is None:
        print(
            "已暂停。进程退出后，使用同一个 --thread 再次运行，"
            "会恢复到这个审批点。"
        )
        return "paused"

    result = graph.invoke(Command(resume=decision), config)
    status = str(result.get("status", "unknown"))
    if status == "saved":
        print(f"已保存报告：{result.get('report_path')}")
    elif status == "cancelled":
        print("已取消保存。")
    elif status == "error":
        print(f"保存失败：{result.get('feedback')}")
    else:
        print(f"流程结束，状态：{status}")
    return status


def run_hitl_cli(
    graph: Any,
    *,
    question: str | None,
    thread_id: str,
    user_id: str,
    decision_mode: str = "ask",
) -> int:
    print("个人知识研究助手 v0.6 HITL")
    print(f"当前 user_id: {user_id}")
    print(f"当前 thread_id: {thread_id}")
    print("流程：生成报告 -> 暂停审批 -> 通过保存 / 拒绝取消")
    print("命令：/new  /exit")

    if question:
        status = run_hitl_question(
            graph,
            question=question,
            thread_id=thread_id,
            user_id=user_id,
            decision_mode=decision_mode,
        )
        return 0 if status != "paused" else 0

    while True:
        try:
            text = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not text:
            continue
        if text in {"/exit", "/quit"}:
            print("再见。")
            break
        if text == "/new":
            import uuid

            thread_id = uuid.uuid4().hex[:8]
            print(f"已新建会话，thread_id: {thread_id}")
            continue

        status = run_hitl_question(
            graph,
            question=text,
            thread_id=thread_id,
            user_id=user_id,
            decision_mode=decision_mode,
        )
        if status == "paused":
            break

    return 0
