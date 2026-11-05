"""Personal Knowledge Agent 命令行入口。

支持三种模式：
- rag：Agentic RAG + Memory（默认）
- basic：基础对话 + Memory
- mcp：通过 MCP Client 调用本地知识库工具

Day 4 默认使用 SQLite 短期记忆和 Store 长期记忆：
- 同一个 ``--thread`` 重启后可以恢复历史；
- 同一个 ``--user`` 的不同 thread 共享偏好；
- ``/remember key=value`` 显式写入偏好，``/preferences`` 查看。

用法：

    uv run python -m app.cli
    uv run python -m app.cli --mode basic
    uv run python -m app.cli --mode mcp --question "MCP 的 stdio 适合什么场景？"
    uv run python -m app.cli --thread day4-chat --user adrian
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage

from app.config import configure_logging, get_settings
from app.graph import build_chat_graph
from app.memory.checkpointer import open_checkpointer
from app.memory.store import (
    load_preferences,
    open_store,
    save_preference,
)
from app.rag.graph import build_rag_graph, initial_rag_state
from app.text import clean_text, content_text

logger = logging.getLogger(__name__)

EXIT_COMMANDS = {"/exit", "/quit"}
NEW_THREAD_COMMANDS = {"/new"}
REMEMBER_PREFIX = "/remember "
PERMISSIONS_COMMANDS = {"/preferences", "/memory"}
AskFn = Callable[[Any, str, str, str], str]


def _configure_stdio() -> None:
    """把 stdin 的坏字节处理方式从 surrogateescape 改成 replace。

    交互式输入里如果出现半截 UTF-8（例如输入法只发出了中文逗号
    ``EF BC 8C`` 的第一个字节 ``EF``），sys.stdin 默认会生成
    ``\\udcef`` 这样的孤立代理项。改成 replace 后会得到 U+FFFD，
    不会在 JSON 编码时崩溃。
    """
    stdin = sys.stdin
    if hasattr(stdin, "reconfigure"):
        try:
            stdin.reconfigure(errors="replace")
        except (ValueError, OSError):
            # pytest/IDE 的 stdin 可能是自定义对象，忽略即可。
            pass


def _clean_question(question: str) -> str:
    """兜底清洗：即使 stdin 没有成功 reconfigure，也不让代理项进入模型请求。"""
    cleaned = clean_text(question)
    if cleaned != question:
        logger.warning("输入包含无法编码的字符，已替换为 U+FFFD")
    return cleaned


def _print_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        print("检索到的来源：无（知识库未检索到可用文档）")
        return
    print("检索到的来源（按相关度）：")
    for source in sources:
        score = source.get("score")
        score_text = (
            f" (score={score:.3f})"
            if isinstance(score, (int, float))
            else ""
        )
        print(f"- {source.get('source', 'unknown')}{score_text}")


def _print_preferences(preferences: dict[str, str]) -> None:
    if not preferences:
        print("当前用户还没有保存长期偏好。")
        return
    print("当前用户长期偏好：")
    for key, value in sorted(preferences.items()):
        print(f"- {key}: {value}")


# ---------------------------------------------------------------------------
# Day 1：基础对话
# ---------------------------------------------------------------------------


def ask_stream(
    graph: Any, question: str, thread_id: str, user_id: str = "default"
) -> str:
    """流式提问并打印 token，返回完整答案。"""
    question = _clean_question(question)
    config = {"configurable": {"thread_id": thread_id}}
    chunks: list[str] = []

    for chunk, metadata in graph.stream(
        {
            "messages": [HumanMessage(content=question)],
            "user_id": user_id,
        },
        config=config,
        stream_mode="messages",
    ):
        # messages 模式会同时回放输入消息和模型输出，这里只打印 agent 节点的 AI 文本。
        if metadata.get("langgraph_node") != "agent":
            continue
        if not isinstance(chunk, AIMessage):
            continue
        text = content_text(chunk.content)
        if text:
            print(text, end="", flush=True)
            chunks.append(text)

    print()
    return "".join(chunks)


def ask_once(
    graph: Any, question: str, thread_id: str, user_id: str = "default"
) -> str:
    """非流式提问，便于脚本调用和对比调试。"""
    question = _clean_question(question)
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "user_id": user_id,
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    answer = content_text(result["messages"][-1].content)
    print(answer)
    return answer


# ---------------------------------------------------------------------------
# Day 2：Agentic RAG
# ---------------------------------------------------------------------------


def ask_rag_stream(
    graph: Any, question: str, thread_id: str, user_id: str = "default"
) -> str:
    """流式执行 RAG，打印答案和来源。"""
    question = _clean_question(question)
    config = {"configurable": {"thread_id": thread_id}}
    chunks: list[str] = []

    for chunk, metadata in graph.stream(
        initial_rag_state(question, user_id=user_id),
        config=config,
        stream_mode="messages",
    ):
        # grade / rewrite 也可能调用模型；只展示 generate 节点的 token。
        if metadata.get("langgraph_node") != "generate":
            continue
        if not isinstance(chunk, AIMessage):
            continue
        text = content_text(chunk.content)
        if text:
            print(text, end="", flush=True)
            chunks.append(text)

    print()
    snapshot = graph.get_state(config)
    values = snapshot.values if snapshot else {}
    if not chunks:
        messages = values.get("messages", [])
        if messages:
            print(content_text(messages[-1].content))
    _print_sources(values.get("sources", []))
    return "".join(chunks)


def ask_rag_once(
    graph: Any, question: str, thread_id: str, user_id: str = "default"
) -> str:
    """非流式执行 RAG。"""
    question = _clean_question(question)
    result = graph.invoke(
        initial_rag_state(question, user_id=user_id),
        config={"configurable": {"thread_id": thread_id}},
    )
    answer = content_text(result["messages"][-1].content)
    print(answer)
    _print_sources(result.get("sources", []))
    return answer


# ---------------------------------------------------------------------------
# 交互循环
# ---------------------------------------------------------------------------


def chat_loop(
    graph: Any,
    thread_id: str,
    ask_fn: AskFn,
    *,
    mode: str,
    store: Any,
    user_id: str,
) -> None:
    title = (
        "v0.4 Agentic RAG + Memory"
        if mode == "rag"
        else "v0.4 基础对话 + Memory"
    )
    print(f"个人知识研究助手 {title}")
    print(f"当前 user_id: {user_id}")
    print(f"当前 thread_id: {thread_id}")
    print("命令：/remember key=value  /preferences  /new  /exit")

    while True:
        try:
            question = _clean_question(input("\n你> ").strip())
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not question:
            continue
        if question in EXIT_COMMANDS:
            print("再见。")
            break
        if question in NEW_THREAD_COMMANDS:
            thread_id = uuid.uuid4().hex[:8]
            print(f"已新建会话，thread_id: {thread_id}")
            continue
        if question.startswith(REMEMBER_PREFIX):
            payload = question[len(REMEMBER_PREFIX) :].strip()
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
            _print_preferences(preferences)
            continue
        if question in PERMISSIONS_COMMANDS:
            _print_preferences(load_preferences(store, user_id))
            continue

        print("助手> ", end="", flush=True)
        ask_fn(graph, question, thread_id, user_id)


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Personal Knowledge Agent - Day 4 CLI"
    )
    parser.add_argument("-q", "--question", help="只问一个问题然后退出")
    parser.add_argument(
        "--thread", default="cli-default", help="会话 thread_id，多轮对话复用"
    )
    parser.add_argument(
        "--user", default=settings.default_user, help="用户 id，长期记忆按用户隔离"
    )
    parser.add_argument(
        "--mode",
        choices=["rag", "basic", "mcp"],
        default="rag",
        help="rag=Agentic RAG（默认），basic=基础对话，mcp=MCP 知识库工具",
    )
    parser.add_argument(
        "--checkpoint-db",
        default=settings.checkpoint_db,
        help="SQLite checkpoint 数据库路径",
    )
    parser.add_argument(
        "--memory-db",
        default=settings.memory_db,
        help="SQLite 长期记忆数据库路径",
    )
    parser.add_argument("--no-stream", action="store_true", help="关闭 token 流式输出")
    parser.add_argument("--log-level", help="日志级别，例如 DEBUG/INFO/WARNING")
    args = parser.parse_args(argv)

    _configure_stdio()
    configure_logging(args.log_level)

    if args.mode in {"rag", "mcp"} and not Path(settings.chroma_dir).exists():
        print(
            "提示：知识库索引不存在。请先运行：\n"
            "  uv run python -m app.rag.ingest"
        )

    if args.mode == "mcp":
        # MCP 工具和 ClientSession 是异步 API；MCP 模式使用 AsyncSqliteSaver。
        from app.mcp.agent import run_mcp_mode

        logger.info(
            "cli.mcp user=%s thread=%s checkpoint_db=%s memory_db=%s",
            args.user,
            args.thread,
            args.checkpoint_db,
            args.memory_db,
        )
        with open_store(args.memory_db) as store:
            return asyncio.run(
                run_mcp_mode(
                    question=args.question,
                    thread_id=args.thread,
                    user_id=args.user,
                    store=store,
                    checkpoint_db=args.checkpoint_db,
                    stream=not args.no_stream,
                )
            )

    with open_store(args.memory_db) as store, open_checkpointer(
        args.checkpoint_db
    ) as checkpointer:
        if args.mode == "rag":
            graph = build_rag_graph(
                checkpointer=checkpointer, store=store
            )
            ask_fn = ask_rag_once if args.no_stream else ask_rag_stream
        else:
            graph = build_chat_graph(
                checkpointer=checkpointer, store=store
            )
            ask_fn = ask_once if args.no_stream else ask_stream

        logger.info(
            "cli.mode=%s user=%s thread=%s checkpoint_db=%s memory_db=%s",
            args.mode,
            args.user,
            args.thread,
            args.checkpoint_db,
            args.memory_db,
        )

        if args.question:
            ask_fn(graph, args.question, args.thread, args.user)
            return 0

        chat_loop(
            graph,
            args.thread,
            ask_fn,
            mode=args.mode,
            store=store,
            user_id=args.user,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
