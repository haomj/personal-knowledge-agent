# Personal Knowledge Research Agent

按 10 天计划逐步实现一个可部署的个人知识研究助手。当前进度：**Day 3 — v0.3-memory**。

## 当前能力

### Day 1：基础 Agent

- LangGraph 最小图：`START -> agent -> END`
- DeepSeek `deepseek-flash`（OpenAI 兼容接口）
- 命令行多轮对话、逐 token 流式输出、模型调用日志

### Day 2：Agentic RAG

- 加载 Markdown / TXT 文档，递归切分 chunk
- DashScope `text-embedding-v4` 生成 1024 维向量
- Chroma 本地向量库，cosine 距离
- 检索结果相关性评分（LLM + function calling）
- 检索不足时改写查询并重试一次
- 回答带 `[来源: 文件路径]` 引用，CLI 单独展示检索来源

### Day 3：Memory 与持久化

- **短期记忆**：SQLite checkpointer，按 `thread_id` 保存会话状态
- **重启恢复**：进程退出后，用同一个 `thread_id` 继续对话
- **长期记忆**：LangGraph `SqliteStore` 保存用户偏好
- **跨 thread 共享**：同一个 `user_id` 的不同 thread 共享偏好
- **用户隔离**：不同 `user_id` 使用不同 namespace，互不可见
- CLI 命令：`/remember key=value`、`/preferences`、`/new`

```text
短期记忆：
Thread -> Conversation -> Checkpoint -> SQLite

长期记忆：
User -> /remember -> Store(namespace=users/<user_id>/preferences)
     -> 下一轮生成时注入 system prompt
```

## 环境准备

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync
```

复制 `.env.example` 为 `.env`，填入真实 Key：

```dotenv
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-flash
LLM_TEMPERATURE=0.7
LLM_THINKING=disabled

DASHSCOPE_API_KEY=your-dashscope-api-key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v4

DOCS_DIR=data/docs
CHROMA_DIR=data/chroma
CHROMA_COLLECTION=knowledge_base
RAG_TOP_K=4
RAG_MAX_RETRIES=1

CHECKPOINT_DB=data/checkpoints.sqlite
MEMORY_DB=data/memory.sqlite
DEFAULT_USER=local-user
```

> `.env`、`data/chroma/`、`data/*.sqlite*` 都已被 `.gitignore` 忽略，不要提交真实 Key 或本地会话数据。

## 构建知识库索引

```bash
uv run python -m app.rag.ingest
```

或者：

```bash
uv run personal-knowledge-ingest
```

默认加载 `data/docs/`，切分后重建 `data/chroma/`。加 `--append` 可以不清空旧索引直接追加。

## 运行

默认使用 Agentic RAG + Memory：

```bash
uv run python -m app.cli
```

指定用户和会话：

```bash
uv run python -m app.cli --user adrian --thread day3-chat
```

单轮提问：

```bash
uv run python -m app.cli --question "Agentic RAG 最多重试几次？" --user adrian
```

回到基础对话模式：

```bash
uv run python -m app.cli --mode basic --user adrian
```

## 记忆命令

在交互式 CLI 中：

```text
/remember project_code=ZK-2026
/preferences
/new
/exit
```

- `/remember key=value`：把偏好写入当前用户的 Store namespace。
- `/preferences`：查看当前用户的全部长期偏好。
- `/new`：新建一个 thread；长期偏好仍然共享。

## 验证重启恢复

第一个进程：

```bash
printf '我叫 Adrian，请记住。\n/exit\n' | \
  uv run python -m app.cli --mode basic --thread day3-persist --user adrian
```

第二个进程，使用同一个 `--thread`：

```bash
printf '我叫什么名字？只回答名字。\n/exit\n' | \
  uv run python -m app.cli --mode basic --thread day3-persist --user adrian
```

第二个进程会从 SQLite checkpoint 恢复第一轮历史，回答 `Adrian`。

## 验证长期记忆隔离

给 adrian 保存一个无法从问题本身猜到的偏好：

```bash
printf '/remember project_code=ZK-2026\n/exit\n' | \
  uv run python -m app.cli --mode basic --thread day3-code-a --user adrian
```

同一用户、不同 thread：

```bash
printf '我的项目代号是什么？\n/exit\n' | \
  uv run python -m app.cli --mode basic --thread day3-code-b --user adrian
```

不同用户：

```bash
printf '我的项目代号是什么？\n/exit\n' | \
  uv run python -m app.cli --mode basic --thread day3-code-c --user bob
```

预期：adrian 回答 `ZK-2026`；bob 表示无法确定。

## 测试

离线测试（不访问网络）：

```bash
uv run pytest
```

真实 Embedding + DeepSeek 的 RAG 端到端测试：

```bash
RUN_LIVE_TESTS=1 uv run pytest tests/test_rag_live.py -q
```

Day 1 的真实模型冒烟测试：

```bash
RUN_LIVE_TESTS=1 uv run pytest tests/test_smoke_live.py -q
```

## 目录

```text
app/
├── config.py            # 环境变量、模型/Embedding 工厂、日志
├── state.py             # AgentState（messages + RAG + user_id）
├── graph.py             # 基础图，支持 Store 偏好注入
├── entrypoint.py        # LangGraph Server 加载入口
├── cli.py               # 命令行：RAG / 基础对话 / 记忆命令
├── text.py              # 文本清洗
├── rag/
│   ├── loader.py        # Markdown/TXT 加载
│   ├── splitter.py      # 递归切分
│   ├── embeddings.py    # Embedding 工厂
│   ├── vector_store.py  # Chroma 创建/重建
│   ├── retriever.py     # 相似度检索
│   ├── evaluator.py     # Retrieval Grading
│   ├── rewriter.py      # Query Rewriting
│   ├── graph.py         # Agentic RAG 图
│   └── ingest.py        # 索引构建脚本
└── memory/
    ├── checkpointer.py  # SQLite 短期记忆
    └── store.py         # SQLite 长期记忆（用户偏好）

data/
├── docs/                # 10 篇本地技术文档（会提交）
├── chroma/              # 生成的向量索引（不提交）
├── checkpoints.sqlite   # 会话 checkpoint（不提交）
└── memory.sqlite        # 用户长期偏好（不提交）

tests/
├── test_graph.py        # Day 1 基础图
├── test_text.py         # 输入清洗
├── test_rag_pipeline.py # 加载/切分/Chroma
├── test_rag_graph.py    # RAG 路由
├── test_memory.py       # SQLite 重启恢复/偏好共享/用户隔离
├── test_smoke_live.py   # Day 1 真实模型
└── test_rag_live.py     # Day 2 真实 RAG

docs/
├── day01-langgraph.md
├── day02-agentic-rag.md
└── day03-memory.md
```

## 开发进度

- [x] Day 1：LangGraph 骨架 + 基础 Agent
- [x] Day 2：Agentic RAG
- [x] Day 3：Memory 与持久化
- [ ] Day 4：MCP
- [ ] Day 5：MultiAgent
- [ ] Day 6：HITL
- [ ] Day 7：Middleware
- [ ] Day 8：Runtime API 与 Web UI
- [ ] Day 9：Docker 部署
- [ ] Day 10：端到端测试与交付
