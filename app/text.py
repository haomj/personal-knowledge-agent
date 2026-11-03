"""文本清洗工具。

终端或输入法可能在某个字符只发送了一半 UTF-8 字节时，
让 Python 用 surrogateescape 解出孤立代理项（如 ``\\udcef``）。
这类字符无法编码成 JSON/UTF-8，送到模型 API 会直接抛
``UnicodeEncodeError: surrogates not allowed``。
"""

from __future__ import annotations

from typing import Any


def clean_text(text: str) -> str:
    """把孤立代理项替换为 U+FFFD，其余字符保持原样。"""
    if not text:
        return text
    return "".join(
        "�" if 0xD800 <= ord(char) <= 0xDFFF else char for char in text
    )


def clean_content(content: Any) -> Any:
    """递归清洗消息 content（支持 str、content blocks 和嵌套 dict）。"""
    if isinstance(content, str):
        return clean_text(content)
    if isinstance(content, list):
        return [clean_content(item) for item in content]
    if isinstance(content, dict):
        return {key: clean_content(value) for key, value in content.items()}
    return content
