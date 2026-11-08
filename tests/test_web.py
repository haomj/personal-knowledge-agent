"""Day 8 Web Runtime 测试：静态页、SSE、线程状态、HITL API。"""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.hitl.graph import build_hitl_graph
from app.web.server import create_app


class FakeRagGraph:
    def stream(self, state, config=None, stream_mode=None):
        yield (
            "messages",
            (
                AIMessage(content="你好，这是流式回答。"),
                {"langgraph_node": "generate"},
            ),
        )
        yield (
            "updates",
            {
                "retrieve": {
                    "documents": [1, 2],
                    "sources": [
                        {
                            "source": "data/docs/03-agentic-rag.md",
                            "score": 0.9,
                        }
                    ],
                }
            },
        )
        yield (
            "updates",
            {
                "grade": {
                    "retrieval_sufficient": True,
                    "grade_reason": "测试资料足够",
                }
            },
        )

    def get_state(self, config):
        return SimpleNamespace(
            values={
                "sources": [
                    {
                        "source": "data/docs/03-agentic-rag.md",
                        "score": 0.9,
                    }
                ]
            }
        )


def _draft(question: str) -> tuple[str, str]:
    return (
        "Web 测试报告",
        f"关于「{question}」的正文。\n[来源: data/docs/03-agentic-rag.md]",
    )


def _make_hitl_graph(tmp_path):
    return build_hitl_graph(
        draft_fn=_draft,
        checkpointer=InMemorySaver(),
        reports_dir=tmp_path / "reports",
    )


def test_web_health_and_static_page(tmp_path):
    app = create_app(
        rag_graph=FakeRagGraph(),
        hitl_graph=_make_hitl_graph(tmp_path),
    )
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        page = client.get("/")
        assert page.status_code == 200
        assert "Personal Knowledge Agent" in page.text
        assert "HITL" in page.text

        created = client.post("/api/threads")
        assert created.status_code == 200
        assert created.json()["thread_id"]


def test_rag_sse_stream(tmp_path):
    app = create_app(
        rag_graph=FakeRagGraph(),
        hitl_graph=_make_hitl_graph(tmp_path),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/chat/stream",
            json={
                "message": "测试问题",
                "thread_id": "t-rag",
                "user_id": "u1",
            },
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        body = response.text
        assert "流式回答" in body
        assert '"type": "step"' in body
        assert "data/docs/03-agentic-rag.md" in body
        assert '"type": "done"' in body


def test_hitl_start_resume_and_thread_state(tmp_path):
    hitl_graph = _make_hitl_graph(tmp_path)
    app = create_app(rag_graph=FakeRagGraph(), hitl_graph=hitl_graph)
    with TestClient(app) as client:
        started = client.post(
            "/api/hitl/start",
            json={
                "message": "生成一份测试报告",
                "thread_id": "t-hitl",
                "user_id": "u1",
            },
        )
        assert started.status_code == 200
        start_data = started.json()
        assert start_data["status"] == "interrupt"
        assert start_data["interrupt"]["title"] == "Web 测试报告"
        assert "content_preview" in start_data["interrupt"]

        state = client.get(
            "/api/threads/t-hitl?mode=hitl&user_id=u1"
        )
        assert state.status_code == 200
        assert state.json()["interrupt"] is not None

        resumed = client.post(
            "/api/hitl/resume",
            json={
                "thread_id": "t-hitl",
                "user_id": "u1",
                "approved": True,
                "feedback": "Web 审批通过",
            },
        )
        assert resumed.status_code == 200
        resume_data = resumed.json()
        assert resume_data["status"] == "saved"
        assert resume_data["report_path"]

        state = client.get(
            "/api/threads/t-hitl?mode=hitl&user_id=u1"
        )
        assert state.status_code == 200
        assert state.json()["status"] == "saved"
        assert state.json()["interrupt"] is None


def test_hitl_resume_without_pending_returns_409(tmp_path):
    app = create_app(
        rag_graph=FakeRagGraph(),
        hitl_graph=_make_hitl_graph(tmp_path),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/hitl/resume",
            json={
                "thread_id": "missing",
                "user_id": "u1",
                "approved": True,
            },
        )
        assert response.status_code == 409
