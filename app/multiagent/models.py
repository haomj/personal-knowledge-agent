"""MultiAgent 的共享数据结构。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubagentResult:
    """子 Agent 返回给 Supervisor 的结果。

    Supervisor 只看到 answer 和 sources，看不到子 Agent 的完整上下文，
    这就是 Context Isolation。
    """

    agent: str
    query: str
    answer: str
    sources: list[str]

    def to_text(self) -> str:
        source_text = "\n".join(f"- {source}" for source in self.sources)
        if not source_text:
            source_text = "（无来源）"
        return (
            f"[{self.agent}] 子任务：{self.query}\n"
            f"回答：{self.answer}\n"
            f"来源：\n{source_text}"
        )
