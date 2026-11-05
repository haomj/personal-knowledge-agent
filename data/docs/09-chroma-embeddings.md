# Chroma 与 Embedding：本地向量检索

Chroma 是一个轻量级本地向量数据库，适合学习和单实例部署。LangChain 通过 `langchain-chroma` 提供 `Chroma` 封装。

## 基本用法

```python
from langchain_chroma import Chroma

store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="data/chroma",
    collection_name="knowledge_base",
    collection_metadata={"hnsw:space": "cosine"},
)
```

## 维度

集合创建时，向量维度由第一份 embedding 决定。之后写入的向量必须维度一致。换 Embedding 模型时，通常要：

1. 删除旧集合或使用新的集合名；
2. 重新写入所有文档；
3. 更新检索配置。

## 距离度量

Chroma 常见距离度量：

- `l2`：欧氏距离，越小越相似；
- `cosine`：余弦距离，适合文本 embedding；
- `ip`：内积。

文本检索通常选择 `cosine`。使用 `similarity_search_with_relevance_scores` 时，分数会转换成 0 到 1 的相关性分数，越大越相关。

## 切分策略

切分太大，检索结果噪声多；切分太小，语义不完整。建议：

- `chunk_size` 在 400 到 1000 字符之间；
- `chunk_overlap` 在 50 到 150 字符之间；
- 优先按标题、段落、句子切分；
- 保留 `source` 元数据，回答时显示来源。

## 持久化

`persist_directory` 指向本地目录。Chroma 会自动持久化。部署时要把这个目录挂载到 volume，否则容器重启后索引丢失。

## 重建索引

文档更新后需要重新切分和写入。可以在 ingestion 脚本里提供 `--reset`，删除旧目录后重建。生产环境应使用增量更新或版本化集合，避免全量重建造成服务中断。
