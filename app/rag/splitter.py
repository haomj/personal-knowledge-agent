"""文档切分：按 Markdown 标题、段落和中文标点递归切分。"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
    *,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> list[Document]:
    """把文档切成适合检索的 chunk，并补充 chunk 元数据。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # 优先保留标题和段落边界，再退到句子和字符。
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    per_source_index: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        index = per_source_index.get(source, 0)
        per_source_index[source] = index + 1
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_id"] = f"{source}#{index}"
    return chunks
