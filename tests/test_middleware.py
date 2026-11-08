"""Day 7 Middleware 单元测试。"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from app.middleware.logging import LoggingMiddleware
from app.middleware.permission import PermissionMiddleware
from app.middleware.retry import RetryMiddleware
from app.middleware.tools import UserContext


def _request(name: str, role: str = "viewer") -> SimpleNamespace:
    return SimpleNamespace(
        tool_call={
            "name": name,
            "args": {"query": "test"},
            "id": "call-1",
            "type": "tool_call",
        },
        runtime=SimpleNamespace(
            context=UserContext(user_id="u1", role=role)
        ),
    )


def _ok() -> ToolMessage:
    return ToolMessage(content="ok", tool_call_id="call-1")


def test_permission_denies_high_risk_for_viewer():
    called = False

    def handler(request):
        nonlocal called
        called = True
        return _ok()

    result = PermissionMiddleware().wrap_tool_call(
        _request("save_research_report", role="viewer"), handler
    )

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "权限拒绝" in result.content
    assert called is False


def test_permission_allows_high_risk_for_admin():
    result = PermissionMiddleware().wrap_tool_call(
        _request("save_research_report", role="admin"),
        lambda request: _ok(),
    )
    assert result.status == "success"


def test_retry_retries_safe_tool_until_success():
    class FlakyHandler:
        def __init__(self):
            self.calls = 0

        def __call__(self, request):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("temporary")
            return _ok()

    handler = FlakyHandler()
    middleware = RetryMiddleware(max_attempts=3, initial_delay=0)
    result = middleware.wrap_tool_call(
        _request("lookup_with_retry"), handler
    )

    assert handler.calls == 3
    assert result.status == "success"


def test_retry_does_not_retry_side_effect_tool():
    class FailingHandler:
        def __init__(self):
            self.calls = 0

        def __call__(self, request):
            self.calls += 1
            raise RuntimeError("write failed")

    handler = FailingHandler()
    middleware = RetryMiddleware(max_attempts=3, initial_delay=0)

    with pytest.raises(RuntimeError):
        middleware.wrap_tool_call(
            _request("save_research_report"), handler
        )

    assert handler.calls == 1


def test_retry_exhausts_and_raises():
    def handler(request):
        raise RuntimeError("always fails")

    middleware = RetryMiddleware(max_attempts=2, initial_delay=0)
    with pytest.raises(RuntimeError):
        middleware.wrap_tool_call(_request("read_knowledge"), handler)


def test_logging_wraps_tool_call(caplog):
    caplog.set_level(logging.INFO, logger="app.middleware.logging")

    result = LoggingMiddleware().wrap_tool_call(
        _request("read_knowledge"), lambda request: _ok()
    )

    assert result.status == "success"
    assert "mw.logging.tool.start" in caplog.text
    assert "mw.logging.tool.done" in caplog.text
    assert "name=read_knowledge" in caplog.text


def test_logging_logs_and_reraises_tool_error(caplog):
    caplog.set_level(logging.INFO, logger="app.middleware.logging")

    def handler(request):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        LoggingMiddleware().wrap_tool_call(
            _request("read_knowledge"), handler
        )

    assert "mw.logging.tool.error" in caplog.text
