"""Logging Middleware：记录模型调用、工具调用、耗时和错误。"""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)


class LoggingMiddleware(AgentMiddleware):
    """最外层 Middleware，负责观察整个模型/工具调用链。"""

    def before_model(self, state: Any, runtime: Any):
        logger.info(
            "mw.logging.before_model messages=%d",
            len(state.get("messages", [])),
        )
        return None

    def after_model(self, state: Any, runtime: Any):
        logger.info(
            "mw.logging.after_model messages=%d",
            len(state.get("messages", [])),
        )
        return None

    def wrap_model_call(self, request: Any, handler):
        started = time.perf_counter()
        logger.info(
            "mw.logging.model.start messages=%d", len(request.messages)
        )
        try:
            response = handler(request)
        except Exception:
            logger.exception(
                "mw.logging.model.error elapsed=%.3fs",
                time.perf_counter() - started,
            )
            raise
        logger.info(
            "mw.logging.model.done elapsed=%.3fs",
            time.perf_counter() - started,
        )
        return response

    def wrap_tool_call(self, request: Any, handler):
        name = request.tool_call.get("name", "unknown")
        started = time.perf_counter()
        logger.info(
            "mw.logging.tool.start name=%s args=%s",
            name,
            request.tool_call.get("args", {}),
        )
        try:
            result = handler(request)
        except Exception:
            logger.exception(
                "mw.logging.tool.error name=%s elapsed=%.3fs",
                name,
                time.perf_counter() - started,
            )
            raise

        status = getattr(result, "status", "success")
        if isinstance(result, ToolMessage):
            preview = str(result.content)[:120]
        else:
            preview = str(result)[:120]
        logger.info(
            "mw.logging.tool.done name=%s status=%s elapsed=%.3fs content=%r",
            name,
            status,
            time.perf_counter() - started,
            preview,
        )
        return result
