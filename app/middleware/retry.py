"""Retry Middleware：只对只读工具做指数退避重试。"""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)

SAFE_TOOL_NAMES = {"read_knowledge", "lookup_with_retry"}


class RetryMiddleware(AgentMiddleware):
    """对只读工具重试；有副作用的工具绝不自动重试。"""

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        initial_delay: float = 0.1,
        backoff_factor: float = 2.0,
        max_delay: float = 1.0,
        safe_tools: set[str] | None = None,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts 必须 >= 1")
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.safe_tools = safe_tools or SAFE_TOOL_NAMES

    def wrap_tool_call(self, request: Any, handler):
        name = request.tool_call.get("name", "unknown")
        if name not in self.safe_tools:
            logger.info(
                "mw.retry.skip name=%s reason=side_effect_or_unknown", name
            )
            return handler(request)

        delay = self.initial_delay
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = handler(request)
            except Exception as exc:
                if attempt >= self.max_attempts:
                    logger.exception(
                        "mw.retry.exhausted name=%s attempt=%d",
                        name,
                        attempt,
                    )
                    raise
                logger.warning(
                    "mw.retry.error name=%s attempt=%d error=%s sleep=%.2fs",
                    name,
                    attempt,
                    exc,
                    delay,
                )
                time.sleep(delay)
                delay = min(delay * self.backoff_factor, self.max_delay)
                continue

            if (
                isinstance(result, ToolMessage)
                and result.status == "error"
                and attempt < self.max_attempts
            ):
                logger.warning(
                    "mw.retry.tool_error name=%s attempt=%d sleep=%.2fs",
                    name,
                    attempt,
                    delay,
                )
                time.sleep(delay)
                delay = min(delay * self.backoff_factor, self.max_delay)
                continue

            logger.info(
                "mw.retry.success name=%s attempt=%d", name, attempt
            )
            return result

        return handler(request)
