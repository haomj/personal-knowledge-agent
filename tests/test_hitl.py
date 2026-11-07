"""Day 6 HITL 测试：审批、拒绝、重启恢复、幂等、保存失败。"""

from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.hitl.graph import build_hitl_graph, initial_hitl_state
from app.hitl.report import save_research_report
from app.memory.checkpointer import open_checkpointer


def _draft(question: str) -> tuple[str, str]:
    return (
        "测试研究报告",
        f"关于「{question}」的正文。\n[来源: data/docs/03-agentic-rag.md]",
    )


def _pending_interrupt(graph, config):
    snapshot = graph.get_state(config)
    for task in snapshot.tasks:
        if task.interrupts:
            return task.interrupts[0]
    return None


def _make_graph(tmp_path: Path, checkpointer=None):
    return build_hitl_graph(
        draft_fn=_draft,
        checkpointer=checkpointer or InMemorySaver(),
        reports_dir=tmp_path / "reports",
    )


def test_hitl_approve_saves_report(tmp_path: Path):
    graph = _make_graph(tmp_path)
    config = {"configurable": {"thread_id": "approve"}}

    first = graph.invoke(
        initial_hitl_state("测试问题", user_id="u1"), config
    )
    assert "__interrupt__" in first

    pending = _pending_interrupt(graph, config)
    assert pending is not None
    assert pending.value["title"] == "测试研究报告"
    assert pending.value["action"] == "save_research_report"

    result = graph.invoke(
        Command(resume={"approved": True, "feedback": "同意"}), config
    )

    assert result["status"] == "saved"
    report_path = Path(result["report_path"])
    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "# 测试研究报告" in text
    assert "data/docs/03-agentic-rag.md" in text


def test_hitl_reject_cancels_without_file(tmp_path: Path):
    graph = _make_graph(tmp_path)
    config = {"configurable": {"thread_id": "reject"}}

    graph.invoke(initial_hitl_state("测试问题"), config)
    result = graph.invoke(
        Command(resume={"approved": False, "feedback": "不要保存"}), config
    )

    assert result["status"] == "cancelled"
    assert not (tmp_path / "reports").exists()


def test_hitl_restart_restores_pending_approval(tmp_path: Path):
    db_path = tmp_path / "checkpoints.sqlite"
    config = {"configurable": {"thread_id": "restart"}}
    reports_dir = tmp_path / "reports"

    with open_checkpointer(db_path) as checkpointer:
        graph = build_hitl_graph(
            draft_fn=_draft,
            checkpointer=checkpointer,
            reports_dir=reports_dir,
        )
        graph.invoke(initial_hitl_state("重启测试"), config)
        assert _pending_interrupt(graph, config) is not None

    # 模拟进程重启：重新打开同一个 SQLite 文件。
    with open_checkpointer(db_path) as checkpointer:
        graph = build_hitl_graph(
            draft_fn=_draft,
            checkpointer=checkpointer,
            reports_dir=reports_dir,
        )
        assert _pending_interrupt(graph, config) is not None
        result = graph.invoke(
            Command(resume={"approved": True, "feedback": "恢复后同意"}),
            config,
        )
        assert result["status"] == "saved"
        assert Path(result["report_path"]).exists()


def test_save_research_report_is_idempotent(tmp_path: Path):
    first = save_research_report("同一标题", "同一正文", tmp_path / "reports")
    second = save_research_report(
        "同一标题", "同一正文", tmp_path / "reports"
    )

    assert first == second
    files = list((tmp_path / "reports").glob("*.md"))
    assert len(files) == 1
    assert first.read_text(encoding="utf-8").count("# 同一标题") == 1


def test_hitl_save_failure_returns_error_status(tmp_path: Path):
    # 把 reports_dir 指向一个文件，mkdir 会失败。
    bad_dir = tmp_path / "not-a-directory"
    bad_dir.write_text("x", encoding="utf-8")

    graph = build_hitl_graph(
        draft_fn=_draft,
        checkpointer=InMemorySaver(),
        reports_dir=bad_dir,
    )
    config = {"configurable": {"thread_id": "save-error"}}
    graph.invoke(initial_hitl_state("保存失败测试"), config)
    result = graph.invoke(
        Command(resume={"approved": True, "feedback": "同意"}), config
    )

    assert result["status"] == "error"
    assert "保存失败" in result["feedback"]
