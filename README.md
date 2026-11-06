# Personal Knowledge Research Agent

按 10 天计划逐步实现一个可部署的个人知识研究助手。当前进度：**Day 5 — v0.5-multiagent**。

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
- 回答带 `[来源: 文件路径]` 引用

### Day 3：Memory 与持久化

- **短期记忆**：SQLite checkpointer，按 `thread_id` 保存会话状态
- **重启恢复**：进程退出后，用同一个 `thread_id` 继续对话
- **长期记忆**：LangGraph `SqliteStore` 保存用户偏好
- **跨 thread 共享**：同一个 `user_id` 的不同 thread 共享偏好
- **用户隔离**：不同 `user_id` 使用不同 namespace

### Day 4：MCP

- 独立 MCP Server（stdio transport）
- 两个工具：`list_documents()`、`search_documents(query, k=4)`
- 复用 Day 2 的 Chroma Retriever
- MCP Client：工具发现、直接调用工具
- `langchain-mcp-adapters`：把 MCP 工具转换成 LangChain Tool
- Agent 使用 `create_agent` 自主选择并调用 MCP 工具

### Day 5：MultiAgent

- **Supervisor**：判断任务类型、委派子任务、汇总结果
- **Knowledge Agent**：通过 MCP 检索本地知识库并总结
- **Research Agent**：通过 Tavily 联网搜索并总结
- **Tool-based Handoff**：Supervisor 用 `ask_knowledge_agent` / `ask_research_agent` 两个工具委派
- **Context Isolation**：子 Agent 只拿子问题，不共享 Supervisor 完整对话
- **Result Aggregation**：Supervisor 汇总本地来源和网页 URL
- **任务预算**：每次问题最多 3 次子任务、单个 Agent 最多 2 次

```text
用户问题
  -> Supervisor
     ├─ ask_knowledge_agent -> MCP -> Chroma
     └─ ask_research_agent  -> Tavily
  -> Supervisor 汇总
  -> 最终答案（本地来源 + 网页 URL）
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

TAVILY_API_KEY=your-tavily-api-key

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

默认加载 `data/docs/`，切分后重建 `data/chroma/`。

## 运行

默认使用 Agentic RAG + Memory：

```bash
uv run python -m app.cli
```

MultiAgent 模式：

```bash
uv run python -m app.cli --mode multi --user adrian --thread day5-multi
```

单轮 MultiAgent 提问：

```bash
uv run python -m app.cli --mode multi \
  --question "结合本地知识库中 Agentic RAG 的重试策略，以及 2026 年 LangGraph 的最新版本信息，给我一个简短总结。" \
  --no-stream
```

MCP 模式：

```bash
uv run python -m app.cli --mode mcp --user adrian --thread day4-mcp
```

基础对话模式：

```bash
uv run python -m app.cli --mode basic --user adrian
```

### MCP 原始 Client 演示

```bash
uv run python -m app.mcp.client
```

### 查看 MCP Server 日志

默认只输出 WARNING，避免干扰 stdio JSON-RPC。需要排查时：

```bash
MCP_LOG_LEVEL=INFO uv run python -m app.cli --mode mcp --question "MCP 有哪些传输方式？"
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

```bash
printf '我叫 Adrian，请记住。\n/exit\n' | \
  uv run python -m app.cli --mode basic --thread day3-persist --user adrian
```

```bash
printf '我叫什么名字？只回答名字。\n/exit\n' | \
  uv run python -m app.cli --mode basic --thread day3-persist --user adrian
```

第二个进程会从 SQLite checkpoint 恢复第一轮历史，回答 `Adrian`。

MCP/MultiAgent 模式同样支持跨进程恢复，内部使用 `AsyncSqliteSaver`。

## 测试

离线测试（不访问模型/Embedding/Tavily API）：

```bash
uv run pytest
```

MultiAgent 真实端到端测试：

```bash
RUN_LIVE_TESTS=1 uv run pytest tests/test_multiagent_live.py -q
```

MCP 真实端到端测试：

```bash
RUN_LIVE_TESTS=1 uv run pytest tests/test_mcp_live.py -q
```

RAG 真实端到端测试：

```bash
RUN_LIVE_TESTS=1 uv run pytest tests/test_rag_live.py -q
```

## 目录

```text
app/
├── config.py            # 环境变量、模型/Embedding 工厂、日志
├── state.py             # AgentState（messages + RAG + user_id）
├── graph.py             # 基础图，支持 Store 偏好注入
├── entrypoint.py        # LangGraph Server 加载入口
├── cli.py               # rag / basic / mcp / multi 四种模式
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
├── mcp/
│   ├── server.py        # stdio MCP Server
│   ├── client.py        # 原始 MCP Client + LangChain 工具加载
│   └── agent.py         # MCP 工具调用 Agent
├── multiagent/
│   ├── knowledge.py     # Knowledge Agent（MCP 本地知识库）
│   ├── research.py      # Research Agent（Tavily）
│   ├── supervisor.py    # Supervisor + 任务预算 + 汇总
│   └── models.py        # SubagentResult
└── memory/
    ├── checkpointer.py  # SQLite 同步/异步短期记忆
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
├── test_mcp.py          # MCP 工具发现/list_documents
├── test_multiagent.py   # 任务预算/子 Agent
├── test_smoke_live.py   # Day 1 真实模型
├── test_rag_live.py     # Day 2 真实 RAG
├── test_mcp_live.py     # Day 4 真实 MCP
└── test_multiagent_live.py # Day 5 真实 MultiAgent

docs/
├── day01-langgraph.md
├── day02-agentic-rag.md
├── day03-memory.md
├── day04-mcp.md
└── day05-multiagent.md
```

## 开发进度

- [x] Day 1：LangGraph 骨架 + 基础 Agent
- [x] Day 2：Agentic RAG
- [x] Day 3：Memory 与持久化
- [x] Day 4：MCP
- [x] Day 5：MultiAgent
- [ ] Day 6：HITL
- [ ] Day 7：Middleware
- [ ] Day 8：Runtime API 与 Web UI
- [ ] Day 9：Docker 部署
- [ ] Day 10：端到端测试与交付
