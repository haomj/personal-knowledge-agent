"""长期记忆：使用 LangGraph Store 保存用户偏好。

第一版不做自动记忆提取，只通过显式命令（CLI 的 /remember）写入。
namespace 使用 (users, user_id, preferences)，把不同用户隔离。
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from langgraph.store.base import BaseStore
from langgraph.store.sqlite import SqliteStore

from app.text import clean_text

PREFERENCES_KEY = "preferences"


@contextmanager
def open_store(db_path: str | Path) -> Iterator[SqliteStore]:
    """打开一个 SQLite Store，用于跨会话保存用户偏好。"""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteStore.from_conn_string(str(path)) as store:
        yield store


def _safe_user_id(user_id: str | None) -> str:
    """namespace label 不能包含 '.'，空值回退到 default。"""
    value = clean_text(user_id or "default").strip() or "default"
    return value.replace(".", "_")


def user_namespace(user_id: str | None) -> tuple[str, str, str]:
    """每个用户一个独立 namespace。"""
    return ("users", _safe_user_id(user_id), "preferences")


def load_preferences(store: BaseStore, user_id: str | None) -> dict[str, str]:
    """读取当前用户的全部长期偏好。"""
    item = store.get(user_namespace(user_id), PREFERENCES_KEY)
    if item is None or not isinstance(item.value, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in item.value.items()
        if key is not None
    }


def save_preference(
    store: BaseStore,
    user_id: str | None,
    key: str,
    value: str,
) -> dict[str, str]:
    """显式保存一条用户偏好，返回保存后的完整偏好。"""
    key = clean_text(key).strip()
    value = clean_text(value).strip()
    if not key:
        raise ValueError("偏好名称不能为空")

    preferences = load_preferences(store, user_id)
    preferences[key] = value
    store.put(user_namespace(user_id), PREFERENCES_KEY, preferences)
    return preferences


def delete_preference(
    store: BaseStore, user_id: str | None, key: str
) -> dict[str, str]:
    """删除一条偏好；key 不存在时保持原样。"""
    preferences = load_preferences(store, user_id)
    preferences.pop(key, None)
    store.put(user_namespace(user_id), PREFERENCES_KEY, preferences)
    return preferences


def format_preferences(preferences: dict[str, str]) -> str:
    """把偏好格式化成可以拼进 system prompt 的文本。"""
    if not preferences:
        return ""
    lines = ["用户长期偏好（来自 Store）："]
    for key, value in sorted(preferences.items()):
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def preference_prompt(store: BaseStore | None, user_id: str | None) -> str:
    """供节点调用：有 store 时返回偏好文本，否则返回空串。"""
    if store is None:
        return ""
    return format_preferences(load_preferences(store, user_id))
