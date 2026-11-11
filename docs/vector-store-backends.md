# 向量库后端：Chroma / Milvus

> 版本：v1.1.0  
> 默认后端：Chroma  
> 可选后端：Milvus Lite 或远程 Milvus Server  
> 切换方式：`VECTOR_STORE=chroma|milvus`

## 1. 目标

在不改动 Agentic RAG 上层逻辑的前提下，让向量库可切换：

- 保留原有 Chroma 实现；
- 新增 Milvus 实现；
- 通过配置选择后端；
- 检索器、RAG 图、MCP Server、Web Runtime 不需要感知底层向量库。

## 2. 配置

```dotenv
VECTOR_STORE=chroma

# Chroma
CHROMA_DIR=data/chroma
CHROMA_COLLECTION=knowledge_base

# Milvus
MILVUS_STORE_URI=data/milvus.db
MILVUS_COLLECTION=knowledge_base
MILVUS_TOKEN=
MILVUS_INDEX_TYPE=AUTOINDEX
MILVUS_METRIC_TYPE=COSINE
```

### 2.1 为什么用 `MILVUS_STORE_URI` 而不是 `MILVUS_URI`

`pymilvus` 在 import 时会读取环境变量 `MILVUS_URI`，并校验它必须是：

```text
http[s]://host:port
```

而 Milvus Lite 使用本地文件：

```text
data/milvus.db
```

如果直接设置 `MILVUS_URI=data/milvus.db`，`import langchain_milvus` 时就会报：

```text
ConnectionConfigException: Illegal uri: [data/milvus.db]
```

因此应用使用独立变量 `MILVUS_STORE_URI`：

- 本地 Lite：`data/milvus.db`
- 远程 Server：`http://localhost:19530`

只有选择 Milvus 后端时才懒加载 `langchain_milvus`，Chroma 模式完全不会 import pymilvus。

## 3. Chroma 实现

原有实现保留在 `app/rag/vector_store.py`：

```python
Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="data/chroma",
    collection_name="knowledge_base",
    collection_metadata={"hnsw:space": "cosine"},
)
```

- cosine 距离；
- reset 时删除目录；
- 持久化到 `data/chroma/`。

## 4. Milvus 实现

代码在 `app/rag/milvus_store.py`。

### 4.1 Milvus Lite

```dotenv
VECTOR_STORE=milvus
MILVUS_STORE_URI=data/milvus.db
MILVUS_COLLECTION=knowledge_base
MILVUS_METRIC_TYPE=COSINE
```

`Milvus.from_documents(...)` 会在本地启动 Milvus Lite 并写入 `data/milvus.db`。

### 4.2 远程 Milvus Server

```dotenv
VECTOR_STORE=milvus
MILVUS_STORE_URI=http://localhost:19530
MILVUS_COLLECTION=knowledge_base
MILVUS_TOKEN=
```

如果使用 Zilliz Cloud 或开启了认证的 Milvus：

```dotenv
MILVUS_TOKEN=username:password
```

### 4.3 索引参数

```python
index_params = {
    "index_type": "AUTOINDEX",
    "metric_type": "COSINE",
    "params": {},
}
```

- `AUTOINDEX` 对 Milvus Lite 和大多数 Milvus 部署可用；
- 如果目标 Milvus 版本不支持 `AUTOINDEX`，可通过 `MILVUS_INDEX_TYPE` 改为 `FLAT` 等；
- 文本检索默认 `COSINE`，与 Chroma 的 cosine 语义接近。

## 5. 切换后端

### 5.1 建索引

Chroma：

```bash
uv run python -m app.rag.ingest --backend chroma
```

Milvus Lite：

```bash
uv run python -m app.rag.ingest --backend milvus --milvus-uri data/milvus.db
```

或只改 `.env`：

```dotenv
VECTOR_STORE=milvus
MILVUS_STORE_URI=data/milvus.db
```

然后：

```bash
uv run python -m app.rag.ingest
```

### 5.2 运行

```bash
VECTOR_STORE=milvus uv run python -m app.cli --question "Agentic RAG 最多重试几次？"
```

Web：

```bash
VECTOR_STORE=milvus uv run personal-knowledge-web
```

MCP/MultiAgent 会通过 `build_default_retriever()` 自动使用当前后端。

### 5.3 重要：索引相互独立

Chroma 和 Milvus 的索引**不能共用**：

- Chroma：`data/chroma/`
- Milvus：`data/milvus.db`

切换后端后必须重新 ingest，否则会看到空索引/“资料不足”。

## 6. 代码结构

```text
app/rag/
├── vector_store.py    # 工厂：get/build_configured_vector_store
├── milvus_store.py    # Milvus Lite/远程实现
├── retriever.py       # 统一 KnowledgeRetriever
└── ingest.py          # --backend chroma|milvus
```

### 6.1 工厂

```python
def get_configured_vector_store(*, embeddings, settings=None):
    if backend == "milvus":
        return get_milvus_store(...)
    return get_vector_store(...)  # Chroma
```

### 6.2 空索引处理

Milvus 在空集合上直接查询会报：

```text
No index params provided. Could not determine relevance function.
```

`KnowledgeRetriever.retrieve()` 先通过 collection stats 检查行数：

- 0 行：返回空列表，由 RAG 评分节点判定“资料不足”；
- 大于 0：正常检索。

Milvus 的行数通过：

```python
client.get_collection_stats(collection_name=...)
```

获取，兼容 Milvus Lite 的 `_MilvusClientCollection`（没有 `num_entities` 属性）。

## 7. Docker

默认 Chroma：

```dotenv
VECTOR_STORE=chroma
```

Milvus Lite：

```dotenv
VECTOR_STORE=milvus
MILVUS_STORE_URI=/app/data/milvus.db
MILVUS_COLLECTION=knowledge_base
MILVUS_METRIC_TYPE=COSINE
```

```bash
docker compose run --rm agent python -m app.rag.ingest --backend milvus
```

```bash
docker compose up -d
```

`pka-data` volume 会持久化 `milvus.db`。

远程 Milvus：

```dotenv
VECTOR_STORE=milvus
MILVUS_STORE_URI=http://host.docker.internal:19530
MILVUS_TOKEN=
```

> 如果 Milvus Server 跑在宿主机，容器里用 `host.docker.internal`（Docker Desktop）。

## 8. 测试与验证

新增测试：

```bash
uv run pytest tests/test_vector_store_backends.py -q
```

结果：

```text
4 passed
```

覆盖：

1. Chroma 后端写入/检索；
2. Milvus Lite 后端写入/检索；
3. 空 Milvus 索引返回空结果；
4. 非法 `VECTOR_STORE` 被拒绝。

完整离线测试：

```text
50 passed, 6 skipped
```

真实 Embedding 手工验证：

```text
# Milvus
rag.retrieve ... docs=4 top_score=0.879
来源: data/docs/03-agentic-rag.md

# Chroma
rag.retrieve ... docs=4 top_score=0.727
来源: data/docs/03-agentic-rag.md
```

两个后端都能回答同一个问题，Milvus 的 cosine 分数与 Chroma 略有差异，但来源一致。

## 9. 已知问题

1. **Milvus Lite 的 gRPC 日志**  
   启动/写入时可能出现：

   ```text
   grpc._server | Method not implemented! AllocTimestamp
   ```

   这是 Milvus Lite 的已知噪声，索引仍然成功；可通过日志级别过滤。

2. **Milvus Lite 单进程**  
   本地文件适合开发和单实例演示；多实例/生产应使用远程 Milvus Server。

3. **索引参数差异**  
   不同 Milvus 版本对 `AUTOINDEX` 支持不同；必要时设置 `MILVUS_INDEX_TYPE=FLAT` 或目标版本支持的索引。

4. **切换后必须重建索引**  
   Chroma 和 Milvus 索引独立；只改 `VECTOR_STORE` 不会自动迁移数据。

5. **没有做 Milvus 与 Chroma 的检索质量对比评估**  
   当前只验证可运行和来源一致，尚未构建 Recall@K/faithfulness 评估集。

## 10. 后续改进

- Milvus 远程 Server 的 Docker Compose 服务（etcd + MinIO + Milvus）。
- 混合检索（dense + BM25）与 rerank。
- 索引迁移工具：Chroma -> Milvus。
- 评估不同 metric/index 的召回质量。
- 多租户 collection/partition 设计。
