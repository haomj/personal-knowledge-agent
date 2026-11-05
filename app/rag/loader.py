"""文档加载：Day 2 只支持 Markdown 和 TXT。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from app.text import clean_text

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}


def _extract_title(text: str) -> str:
    """取第一个一级标题作为文档标题；没有标题时返回空字符串。"""
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _display_source(path: Path, root: Path) -> str:
    """生成稳定的来源路径：优先相对项目根目录，其次相对文档目录。"""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.relative_to(root).as_posix()


def load_documents(docs_dir: str | Path) -> list[Document]:
    """递归加载目录下的 Markdown/TXT 文件。

    每个 Document 保留以下 metadata：
    - source：用于回答时显示来源；
    - title：第一个一级标题；
    - relative_source：相对 docs_dir 的路径；
    - suffix：文件扩展名。
    """
    root = Path(docs_dir)
    if not root.exists():
        raise FileNotFoundError(f"文档目录不存在: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"不是目录: {root}")

    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = clean_text(path.read_text(encoding="utf-8"))
        if not text.strip():
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": _display_source(path, root),
                    "relative_source": path.relative_to(root).as_posix(),
                    "title": _extract_title(text) or path.stem,
                    "suffix": path.suffix.lower(),
                },
            )
        )

    if not documents:
        raise ValueError(
            f"在 {root} 下没有找到可加载的 Markdown/TXT 文档"
        )
    return documents
