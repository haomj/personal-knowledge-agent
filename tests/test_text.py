"""输入清洗的回归测试。"""

from app.text import clean_content, clean_text


def test_clean_text_replaces_lone_surrogate():
    assert clean_text("你好\udcef，我是 Adrian") == "你好�，我是 Adrian"


def test_clean_text_keeps_valid_text():
    text = "你好，我是 Adrian"
    assert clean_text(text) == text


def test_clean_content_handles_text_blocks():
    content = [
        {"type": "text", "text": "半截字节\udcef"},
        "普通字符串",
    ]
    assert clean_content(content) == [
        {"type": "text", "text": "半截字节�"},
        "普通字符串",
    ]
