"""Day 8：FastAPI Runtime + 最小 Web Chat UI。

启动：

    uv run personal-knowledge-web
    # 或
    uv run uvicorn app.web.server:app --host 127.0.0.1 --port 8000

接口：
- GET  /                     聊天页面
- GET  /api/health           健康检查
- POST /api/threads          生成 thread_id
- GET  /api/threads/{id}     查看线程消息/来源/待审批
- POST /api/chat/stream      RAG 流式对话（SSE）
- POST /api/hitl/start       HITL 开始/恢复待审批
- POST /api/hitl/resume      审批通过/拒绝
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.config import get_settings
from app.hitl.graph import build_hitl_graph, initial_hitl_state
from app.memory.checkpointer import open_checkpointer
from app.memory.store import open_store
from app.rag.graph import build_rag_graph, initial_rag_state
from app.text import content_text

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    thread_id: str = Field(min_length=1, max_length=120)
    user_id: str = Field(default="web-user", min_length=1, max_length=120)


class HitlResumeRequest(BaseModel):
    thread_id: str
    user_id: str = "web-user"
    approved: bool
    feedback: str = ""


def _thread_key(mode: str, user_id: str, thread_id: str) -> str:
    """把用户和逻辑 thread 组合成 checkpoint 的 thread_id，避免用户串线。"""
    raw = f"{mode}:{user_id}:{thread_id}"
    safe = re.sub(r"[^A-Za-z0-9_.:-]+", "_", raw)
    return safe[:200]


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _pending_payload(snapshot: Any) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    for task in getattr(snapshot, "tasks", ()) or ():
        for interrupt in getattr(task, "interrupts", ()) or ():
            value = getattr(interrupt, "value", None)
            if isinstance(value, dict):
                return value
            return {"value": value}
    return None


def _message_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    elif isinstance(message, ToolMessage):
        role = "tool"
    else:
        role = type(message).__name__

    data: dict[str, Any] = {
        "role": role,
        "content": content_text(message.content),
    }
    if isinstance(message, AIMessage) and message.tool_calls:
        data["tool_calls"] = [
            {"name": call.get("name"), "args": call.get("args", {})}
            for call in message.tool_calls
        ]
    if isinstance(message, ToolMessage):
        data["name"] = message.name
        data["status"] = message.status
    return data


def _step_event(node: str, update: Any) -> dict[str, Any] | None:
    if not isinstance(update, dict):
        return None
    if node == "retrieve":
        documents = update.get("documents", [])
        return {
            "type": "step",
            "node": node,
            "label": f"检索知识库：{len(documents)} 条资料",
        }
    if node == "grade":
        sufficient = bool(update.get("retrieval_sufficient"))
        return {
            "type": "step",
            "node": node,
            "label": "资料足够" if sufficient else "资料不足，准备改写查询",
            "reason": update.get("grade_reason", ""),
        }
    if node == "rewrite":
        return {
            "type": "step",
            "node": node,
            "label": f"改写查询：{update.get('query', '')}",
        }
    if node == "generate":
        return {
            "type": "step",
            "node": node,
            "label": "生成带来源的回答",
        }
    return None


def create_app(
    *,
    rag_graph: Any | None = None,
    hitl_graph: Any | None = None,
    auth_token: str | None = None,
) -> FastAPI:
    """创建 FastAPI 应用；测试时可以注入图，避免真实模型/数据库。"""

    configured_token = (
        auth_token
        if auth_token is not None
        else get_settings().web_auth_token
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if rag_graph is not None and hitl_graph is not None:
            app.state.rag_graph = rag_graph
            app.state.hitl_graph = hitl_graph
            yield
            return

        settings = get_settings()
        with open_store(settings.memory_db) as store, open_checkpointer(
            settings.checkpoint_db
        ) as checkpointer:
            app.state.store = store
            app.state.checkpointer = checkpointer
            app.state.rag_graph = build_rag_graph(
                checkpointer=checkpointer, store=store
            )
            app.state.hitl_graph = build_hitl_graph(
                checkpointer=checkpointer
            )
            yield

    app = FastAPI(
        title="Personal Knowledge Agent",
        version="0.9.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def auth_and_log(request: Request, call_next):
        path = request.url.path
        if configured_token and path.startswith("/api/") and path != "/api/health":
            supplied = request.headers.get("x-api-key", "")
            if not supplied:
                authorization = request.headers.get("authorization", "")
                if authorization.lower().startswith("bearer "):
                    supplied = authorization[7:].strip()
            if supplied != configured_token:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Unauthorized: invalid API token"},
                )

        started = time.perf_counter()
        response = await call_next(request)
        logger.info(
            "web.request method=%s path=%s status=%d elapsed=%.3fs",
            request.method,
            path,
            response.status_code,
            time.perf_counter() - started,
        )
        return response

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health():
        settings = get_settings()
        return {
            "status": "ok",
            "model": settings.model_name,
            "modes": ["rag", "hitl"],
            "auth_required": bool(configured_token),
        }

    @app.post("/api/threads")
    def new_thread():
        return {"thread_id": uuid.uuid4().hex[:12]}

    @app.get("/api/threads/{thread_id}")
    def thread_state(
        thread_id: str,
        request: Request,
        mode: Literal["rag", "hitl"] = Query(default="rag"),
        user_id: str = Query(default="web-user"),
    ):
        graph = (
            request.app.state.rag_graph
            if mode == "rag"
            else request.app.state.hitl_graph
        )
        config = {
            "configurable": {
                "thread_id": _thread_key(mode, user_id, thread_id)
            }
        }
        snapshot = graph.get_state(config)
        values = snapshot.values if snapshot else {}
        messages = [
            _message_dict(message)
            for message in values.get("messages", [])[-40:]
        ]
        return {
            "thread_id": thread_id,
            "mode": mode,
            "user_id": user_id,
            "messages": messages,
            "sources": values.get("sources", []),
            "status": values.get("status", ""),
            "report_path": values.get("report_path", ""),
            "interrupt": _pending_payload(snapshot),
        }

    @app.post("/api/chat/stream")
    def chat_stream(payload: ChatRequest, request: Request):
        graph = request.app.state.rag_graph
        config = {
            "configurable": {
                "thread_id": _thread_key(
                    "rag", payload.user_id, payload.thread_id
                )
            }
        }

        def event_stream():
            yield _sse(
                {
                    "type": "start",
                    "mode": "rag",
                    "thread_id": payload.thread_id,
                }
            )
            try:
                state = initial_rag_state(
                    payload.message, user_id=payload.user_id
                )
                for item in graph.stream(
                    state,
                    config=config,
                    stream_mode=["messages", "updates"],
                ):
                    if not (
                        isinstance(item, tuple) and len(item) == 2
                    ):
                        continue
                    stream_mode, data = item
                    if stream_mode == "messages":
                        chunk, metadata = data
                        if (
                            metadata.get("langgraph_node") == "generate"
                            and isinstance(chunk, AIMessage)
                        ):
                            text = content_text(chunk.content)
                            if text:
                                yield _sse(
                                    {"type": "token", "text": text}
                                )
                    elif stream_mode == "updates" and isinstance(
                        data, dict
                    ):
                        for node, update in data.items():
                            event = _step_event(node, update)
                            if event:
                                yield _sse(event)

                snapshot = graph.get_state(config)
                values = snapshot.values if snapshot else {}
                yield _sse(
                    {
                        "type": "sources",
                        "sources": values.get("sources", []),
                    }
                )
                yield _sse({"type": "done"})
            except Exception as exc:
                logger.exception("web.rag.stream failed")
                yield _sse({"type": "error", "message": str(exc)})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/hitl/start")
    def hitl_start(payload: ChatRequest, request: Request):
        graph = request.app.state.hitl_graph
        config = {
            "configurable": {
                "thread_id": _thread_key(
                    "hitl", payload.user_id, payload.thread_id
                )
            }
        }
        snapshot = graph.get_state(config)
        pending = _pending_payload(snapshot)

        if pending is None:
            graph.invoke(
                initial_hitl_state(
                    payload.message, user_id=payload.user_id
                ),
                config,
            )
            snapshot = graph.get_state(config)
            pending = _pending_payload(snapshot)

        if pending is not None:
            return {
                "status": "interrupt",
                "thread_id": payload.thread_id,
                "interrupt": pending,
            }

        values = snapshot.values if snapshot else {}
        return {
            "status": values.get("status", ""),
            "report_path": values.get("report_path", ""),
        }

    @app.post("/api/hitl/resume")
    def hitl_resume(payload: HitlResumeRequest, request: Request):
        graph = request.app.state.hitl_graph
        config = {
            "configurable": {
                "thread_id": _thread_key(
                    "hitl", payload.user_id, payload.thread_id
                )
            }
        }
        snapshot = graph.get_state(config)
        if _pending_payload(snapshot) is None:
            raise HTTPException(
                status_code=409, detail="该线程没有待审批的 HITL 任务"
            )

        result = graph.invoke(
            Command(
                resume={
                    "approved": payload.approved,
                    "feedback": payload.feedback,
                }
            ),
            config,
        )
        return {
            "status": result.get("status", ""),
            "report_path": result.get("report_path", ""),
            "feedback": result.get("feedback", ""),
        }

    return app


app = create_app()


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.web.server:app",
        host=settings.web_host,
        port=settings.web_port,
        reload=False,
    )


if __name__ == "__main__":
    run()
