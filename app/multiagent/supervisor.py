"""Supervisor：通过工具调用把任务交给 Knowledge/Research Agent，并汇总结果。

采用计划建议的 Supervisor + Tool-based Handoff：
- Supervisor 是一个 create_agent，工具是 ask_knowledge_agent / ask_research_agent；
- 每次子任务只把“子问题”传给子 Agent，实现 Context Isolation；
- Supervisor 收到子 Agent 的答案和来源后做 Result Aggregation。

限制：
- 每次用户问题最多 3 次子任务调用；
- 每个子 Agent 最多 2 次；
- Supervisor 提示词要求最多两轮任务分配。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from app.config import get_model, get_settings
from app.memory.store import load_preferences, preference_prompt, save_preference
from app.multiagent.knowledge import run_knowledge_agent
from app.multiagent.models import SubagentResult
from app.multiagent.research import run_research_agent
from app.text import content_text

logger = logging.getLogger(__name__)

SubagentFn = Callable[[str], Awaitable[SubagentResult]]

SUPERVISOR_INSTRUCTIONS = """你是多 Agent 系统的 Supervisor。你的职责是拆解任务、委派子任务并汇总结果。

可用工具：
- ask_knowledge_agent：向本地知识库 Agent 提问，适用于项目文档、代码笔记、已有技术资料。
- ask_research_agent：向联网 Research Agent 提问，适用于最新版本、新闻、外部资料。

规则：
1. 先判断问题需要本地知识、联网信息，还是两者都需要。
2. 如果两者都需要，尽量在同一轮并行调用两个工具。
3. 最多进行两轮任务分配；总共最多 3 次子任务调用；每个子 Agent 最多 2 次。
4. 子 Agent 只拿到你委派的子问题，不共享完整对话；请把子问题写清楚。
5. 收到结果后由你自己汇总成最终答案。最终答案末尾必须有“来源”列表：至少列出本地文件路径，以及 Research Agent 返回的网页 URL（如果存在）。
6. 子 Agent 没有提供的信息不要编造；信息不足时明确说明。
7. Research Agent 对英文查询效果更好；查询版本或官方发布信息时，可在子问题中保留英文关键词（如 latest release、official docs）。
"""


@dataclass
class TaskBudget:
    """一次用户问题内的子任务预算。"""

    max_tasks: int = 3
    max_per_agent: int = 2
    total: int = 0
    per_agent: dict[str, int] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def acquire(self, agent: str) -> bool:
        async with self.lock:
            if self.total >= self.max_tasks:
                return False
            if self.per_agent.get(agent, 0) >= self.max_per_agent:
                return False
            self.total += 1
            self.per_agent[agent] = self.per_agent.get(agent, 0) + 1
            return True


def _supervisor_prompt(
    system_prompt: str | None,
    store: BaseStore | None,
    user_id: str,
) -> str:
    parts = [
        system_prompt or get_settings().system_prompt,
        SUPERVISOR_INSTRUCTIONS,
    ]
    preferences = preference_prompt(store, user_id)
    if preferences:
        parts.append(preferences)
    return "\n\n".join(part for part in parts if part)


def build_multiagent(
    *,
    model: BaseChatModel | None = None,
    knowledge_agent: SubagentFn | None = None,
    research_agent: SubagentFn | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
    user_id: str = "default",
    max_tasks: int = 3,
    max_per_agent: int = 2,
    system_prompt: str | None = None,
):
    """构建 Supervisor Agent。

    子 Agent 可以通过参数注入，便于离线测试。
    """
    knowledge_fn = knowledge_agent or run_knowledge_agent
    research_fn = research_agent or run_research_agent
    budget = TaskBudget(
        max_tasks=max_tasks, max_per_agent=max_per_agent
    )

    @tool
    async def ask_knowledge_agent(question: str) -> str:
        """向本地知识库 Agent 委派子问题；适用于项目文档和已有技术资料。"""
        if not await budget.acquire("knowledge"):
            return "Knowledge Agent 的任务预算已用完，请基于已有结果汇总。"
        try:
            result = await knowledge_fn(question)
        except Exception as exc:
            logger.warning("knowledge_agent failed: %s", exc)
            return f"Knowledge Agent 执行失败：{exc}"
        return result.to_text()

    @tool
    async def ask_research_agent(question: str) -> str:
        """向联网 Research Agent 委派子问题；适用于最新版本、新闻和外部资料。"""
        if not await budget.acquire("research"):
            return "Research Agent 的任务预算已用完，请基于已有结果汇总。"
        try:
            result = await research_fn(question)
        except Exception as exc:
            logger.warning("research_agent failed: %s", exc)
            return f"Research Agent 执行失败：{exc}"
        return result.to_text()

    prompt = _supervisor_prompt(system_prompt, store, user_id)
    agent = create_agent(
        model or get_model(),
        tools=[ask_knowledge_agent, ask_research_agent],
        system_prompt=prompt,
        checkpointer=checkpointer,
        store=store,
    )
    logger.info(
        "multiagent.compiled user=%s max_tasks=%d max_per_agent=%d",
        user_id,
        max_tasks,
        max_per_agent,
    )
    return agent


def _print_tool_trace(messages: list[Any]) -> None:
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                print(
                    f"[Supervisor] 委派 {call.get('name')} "
                    f"{call.get('args', {})}"
                )
        elif isinstance(message, ToolMessage):
            text = content_text(message.content)
            preview = text[:300] + ("..." if len(text) > 300 else "")
            print(
                f"[Supervisor] {message.name or 'tool'} 返回: {preview}"
            )


def _current_turn_messages(messages: list[Any]) -> list[Any]:
    last_human = None
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            last_human = index
    if last_human is None:
        return messages
    return messages[last_human:]


async def ask_multiagent_stream(
    agent: Any, question: str, thread_id: str
) -> str:
    """流式执行 Supervisor，只打印 Supervisor 的 token。"""
    config = {"configurable": {"thread_id": thread_id}}
    chunks: list[str] = []

    async for chunk, _metadata in agent.astream(
        {"messages": [HumanMessage(content=question)]},
        config=config,
        stream_mode="messages",
    ):
        if not isinstance(chunk, AIMessage):
            continue
        text = content_text(chunk.content)
        if text:
            print(text, end="", flush=True)
            chunks.append(text)

    print()
    snapshot = await agent.aget_state(config)
    if snapshot:
        _print_tool_trace(
            _current_turn_messages(snapshot.values.get("messages", []))
        )
    return "".join(chunks)


async def ask_multiagent_once(
    agent: Any, question: str, thread_id: str
) -> str:
    """非流式执行 Supervisor，先打印委派轨迹，再打印最终答案。"""
    config = {"configurable": {"thread_id": thread_id}}
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )
    messages = result["messages"]
    _print_tool_trace(_current_turn_messages(messages))
    answer = content_text(messages[-1].content)
    print(answer)
    return answer


async def run_multiagent_cli(
    *,
    question: str | None,
    thread_id: str,
    user_id: str,
    store: Any,
    checkpointer: Any,
    stream: bool = True,
) -> int:
    """MultiAgent 模式的交互循环。"""

    def make_agent():
        return build_multiagent(
            checkpointer=checkpointer,
            store=store,
            user_id=user_id,
        )

    async def ask(text: str) -> str:
        # 每次提问重新构建 Supervisor，保证 TaskBudget 是“本次问题”的预算。
        agent = make_agent()
        if stream:
            return await ask_multiagent_stream(agent, text, thread_id)
        return await ask_multiagent_once(agent, text, thread_id)

    if question:
        logger.info(
            "cli.multi.once user=%s thread=%s", user_id, thread_id
        )
        await ask(question)
        return 0

    print("个人知识研究助手 v0.5 MultiAgent")
    print(f"当前 user_id: {user_id}")
    print(f"当前 thread_id: {thread_id}")
    print("子 Agent：Knowledge Agent（本地知识库）/ Research Agent（Tavily）")
    print("命令：/remember key=value  /preferences  /new  /exit")

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
            thread_id = uuid.uuid4().hex[:8]
            print(f"已新建会话，thread_id: {thread_id}")
            continue
        if text.startswith("/remember "):
            payload = text[len("/remember ") :].strip()
            if "=" not in payload:
                print("用法：/remember key=value")
                continue
            key, value = payload.split("=", 1)
            try:
                preferences = save_preference(
                    store, user_id, key.strip(), value.strip()
                )
            except ValueError as exc:
                print(f"保存失败：{exc}")
                continue
            print(f"已记住：{key.strip()} = {value.strip()}")
            for item_key, item_value in sorted(preferences.items()):
                print(f"- {item_key}: {item_value}")
            continue
        if text in {"/preferences", "/memory"}:
            preferences = load_preferences(store, user_id)
            if not preferences:
                print("当前用户还没有保存长期偏好。")
            else:
                print("当前用户长期偏好：")
                for key, value in sorted(preferences.items()):
                    print(f"- {key}: {value}")
            continue

        print("助手> ", end="", flush=True)
        await ask(text)

    return 0


async def run_multiagent_mode(
    *,
    question: str | None,
    thread_id: str,
    user_id: str,
    store: Any,
    checkpoint_db: str,
    stream: bool = True,
) -> int:
    """CLI 的 MultiAgent 模式入口：使用异步 SQLite checkpointer。"""
    from app.memory.checkpointer import open_async_checkpointer

    async with open_async_checkpointer(checkpoint_db) as checkpointer:
        return await run_multiagent_cli(
            question=question,
            thread_id=thread_id,
            user_id=user_id,
            store=store,
            checkpointer=checkpointer,
            stream=stream,
        )


if __name__ == "__main__":
    from app.config import configure_logging
    from app.memory.checkpointer import open_async_checkpointer
    from app.memory.store import open_store

    settings = get_settings()
    configure_logging()

    async def _main() -> None:
        with open_store(settings.memory_db) as store:
            async with open_async_checkpointer(
                settings.checkpoint_db
            ) as checkpointer:
                await run_multiagent_cli(
                    question=None,
                    thread_id="multiagent-cli",
                    user_id=settings.default_user,
                    store=store,
                    checkpointer=checkpointer,
                )

    asyncio.run(_main())
