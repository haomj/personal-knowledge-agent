"""SQLite checkpointer：短期记忆（会话历史）。"""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import AsyncIterator, Iterator

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


@contextmanager
def open_checkpointer(db_path: str | Path) -> Iterator[SqliteSaver]:
    """打开一个同步 SQLite checkpointer。

    SqliteSaver.from_conn_string 会在退出时关闭连接。
    父目录不存在时先创建，避免 sqlite3 报 unable to open database file。
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as checkpointer:
        yield checkpointer


@asynccontextmanager
async def open_async_checkpointer(
    db_path: str | Path,
) -> AsyncIterator[AsyncSqliteSaver]:
    """打开一个异步 SQLite checkpointer。

    MCP Agent 使用 ainvoke/astream，必须配 AsyncSqliteSaver；
    同步的 SqliteSaver 没有实现 aget_tuple/aput，会在运行时抛
    NotImplementedError。
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(path)) as checkpointer:
        yield checkpointer
