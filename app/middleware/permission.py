"""Permission Middleware：按用户角色拦截高风险工具。"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)

HIGH_RISK_TOOLS = {"save_research_report"}


class PermissionMiddleware(AgentMiddleware):
    """高风险工具只允许 admin；其他角色返回 ToolMessage error。"""

    def __init__(self, high_risk_tools: set[str] | None = None):
        self.high_risk_tools = high_risk_tools or HIGH_RISK_TOOLS

    def wrap_tool_call(self, request: Any, handler):
        name = request.tool_call.get("name", "unknown")
        context = getattr(request.runtime, "context", None)
        user_id = getattr(context, "user_id", "unknown")
        role = getattr(context, "role", "viewer")

        if name in self.high_risk_tools and role != "admin":
            logger.warning(
                "mw.permission.deny user=%s role=%s tool=%s",
                user_id,
                role,
                name,
            )
            return ToolMessage(
                content=(
                    f"权限拒绝：{name} 是高风险工具，仅 admin 可调用；"
                    f"当前用户 {user_id} 的角色是 {role}。"
                ),
                tool_call_id=request.tool_call.get("id", ""),
                name=name,
                status="error",
            )

        logger.info(
            "mw.permission.allow user=%s role=%s tool=%s",
            user_id,
            role,
            name,
        )
        return handler(request)
