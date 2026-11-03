"""Day 1 命令行聊天入口。

用法：

    uv run python -m app.cli
    uv run python -m app.cli --question "你好"
    printf '我叫 Adrian\\n我叫什么？\\n/exit\\n' | uv run python -m app.cli
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from app.config import configure_logging
from app.graph import build_chat_graph
from app.text import clean_text

logger = logging.getLogger(__name__)

EXIT_COMMANDS = {"/exit", "/quit"}
NEW_THREAD_COMMANDS = {"/new"}


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


def _content_text(content: Any) -> str:
    """把消息内容统一成可打印文本（兼容 str 和 content blocks）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def ask_stream(graph: Any, question: str, thread_id: str) -> str:
    """流式提问并打印 token，返回完整答案。"""
    question = _clean_question(question)
    config = {"configurable": {"thread_id": thread_id}}
    chunks: list[str] = []

    for chunk, metadata in graph.stream(
        {"messages": [HumanMessage(content=question)]},
        config=config,
        stream_mode="messages",
    ):
        # messages 模式会同时回放输入消息和模型输出，这里只打印 agent 节点的 AI 文本。
        if metadata.get("langgraph_node") != "agent":
            continue
        if not isinstance(chunk, AIMessage):
            continue
        text = _content_text(chunk.content)
        if text:
            print(text, end="", flush=True)
            chunks.append(text)

    print()
    return "".join(chunks)


def ask_once(graph: Any, question: str, thread_id: str) -> str:
    """非流式提问，便于脚本调用和对比调试。"""
    question = _clean_question(question)
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    answer = _content_text(result["messages"][-1].content)
    print(answer)
    return answer


def chat_loop(graph: Any, thread_id: str) -> None:
    print("个人知识研究助手 v0.1（输入 /exit 退出，/new 新建会话）")
    print(f"当前 thread_id: {thread_id}")
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

        print("助手> ", end="", flush=True)
        ask_stream(graph, question, thread_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Personal Knowledge Agent - Day 1 CLI"
    )
    parser.add_argument("-q", "--question", help="只问一个问题然后退出")
    parser.add_argument(
        "--thread", default="day1-cli", help="会话 thread_id，多轮对话复用"
    )
    parser.add_argument("--no-stream", action="store_true", help="关闭 token 流式输出")
    parser.add_argument("--log-level", help="日志级别，例如 DEBUG/INFO/WARNING")
    args = parser.parse_args(argv)

    _configure_stdio()
    configure_logging(args.log_level)
    graph = build_chat_graph()

    if args.question:
        logger.info("cli.once thread_id=%s", args.thread)
        if args.no_stream:
            ask_once(graph, args.question, args.thread)
        else:
            ask_stream(graph, args.question, args.thread)
        return 0

    logger.info("cli.chat thread_id=%s", args.thread)
    chat_loop(graph, args.thread)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
