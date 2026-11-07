# Personal Knowledge Research Agent

按 10 天计划逐步实现一个可部署的个人知识研究助手。当前进度：**Day 6 — v0.6-hitl**。

## 当前能力

### Day 1：基础 Agent

- LangGraph 最小图：`START -> agent -> END`
- DeepSeek `deepseek-flash`（OpenAI 兼容接口）
- 命令行多轮对话、逐 token 流式输出、模型调用日志

### Day 2：Agentic RAG

- Markdown / TXT 文档加载、递归切分
- DashScope `text-embedding-v4` 生成 1024 维向量
- Chroma 本地向量库，cosine 距离
- Retrieval Grading、Query Rewriting、最多一次重试
- 回答带 `[来源: 文件路径]` 引用

### Day 3：Memory 与持久化

- **短期记忆**：SQLite checkpointer，按 `thread_id` 保存会话
- **长期记忆**：LangGraph `SqliteStore` 保存用户偏好
- 跨 thread 共享、按用户隔离、重启恢复

### Day 4：MCP

- stdio MCP Server：`list_documents()`、`search_documents(query, k=4)`
- 复用 Day 2 Retriever
- MCP Client：工具发现、直接调用
- `langchain-mcp-adapters` 接入 Agent

### Day 5：MultiAgent

- Supervisor + Knowledge Agent + Research Agent
- Tool-based Handoff、Context Isolation、Result Aggregation
- Knowledge Agent 走 MCP，Research Agent 走 Tavily
- 每次问题最多 3 次子任务、单个 Agent 最多 2 次

### Day 6：HITL

- 生成研究报告 -> `interrupt()` 暂停 -> 人工审批 -> 保存/取消
- 使用真实 LangGraph Interrupt，不是 `input()` 或 API 标志位
- `Command(resume=...)` 恢复；节点会从头重新执行，因此中断前不能有副作用
- SQLite checkpointer：进程重启后，用同一个 `thread_id` 继续审批
- 报告保存是幂等的：同样标题+正文不会重复生成文件

```text
draft_report -> approval(interrupt)
                  ├─ approved -> save_report -> END
                  └─ rejected -> cancel -> END
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
REPORTS_DIR=data/reports
```

> `.env`、`data/chroma/`、`data/*.sqlite*`、`data/reports/` 都已被 `.gitignore` 忽略。

## 构建知识库索引

```bash
uv run python -m app.rag.ingest
```

或者：

```bash
uv run personal-knowledge-ingest
```

## 运行

默认 Agentic RAG：

```bash
uv run python -m app.cli
```

HITL 模式（交互审批）：

```bash
uv run python -m app.cli --mode hitl --question "总结 Agentic RAG 的重试策略"
```

自动批准/拒绝（方便脚本验收）：

```bash
uv run python -m app.cli --mode hitl \
  --question "总结 Agentic RAG 的重试策略" \
  --hitl-decision approve
```

```bash
uv run python -m app.cli --mode hitl \
  --question "总结 MCP 的两种传输方式" \
  --hitl-decision reject
```

暂停后重启恢复：

```bash
uv run python -m app.cli --mode hitl \
  --question "总结 Memory 的短期与长期记忆" \
  --thread day6-pause \
  --hitl-decision pause
```

```bash
uv run python -m app.cli --mode hitl \
  --thread day6-pause \
  --hitl-decision approve
```

MultiAgent 模式：

```bash
uv run python -m app.cli --mode multi --user adrian --thread day5-multi
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

```bash
MCP_LOG_LEVEL=INFO uv run python -m app.cli --mode mcp --question "MCP 有哪些传输方式？"
```

## HITL 审批方式

`--hitl-decision`：

| 值 | 行为 |
| --- | --- |
| `ask`（默认） | 交互输入 y/n/p |
| `approve` | 自动通过 |
| `reject` | 自动拒绝 |
| `pause` | 打印审批请求后退出，下一次同 thread 恢复 |

报告中会显示标题、正文长度和内容预览。审批通过后写入：

```text
data/reports/<标题>-<内容哈希>.md
```

## 记忆命令

在交互式 CLI 中：

```text
/remember project_code=ZK-2026
/preferences
/new
/exit
```

## 验证重启恢复

```bash
printf '我叫 Adrian，请记住。\n/exit\n' | \
  uv run python -m app.cli --mode basic --thread day3-persist --user adrian
```

```bash
printf '我叫什么名字？只回答名字。\n/exit\n' | \
  uv run python -m app.cli --mode basic --thread day3-persist --user adrian
```

## 测试

离线测试：

```bash
uv run pytest
```

HITL 测试：

```bash
uv run pytest tests/test_hitl.py -q
```

真实 MultiAgent 测试：

```bash
RUN_LIVE_TESTS=1 uv run pytest tests/test_multiagent_live.py -q
```

真实 MCP 测试：

```bash
RUN_LIVE_TESTS=1 uv run pytest tests/test_mcp_live.py -q
```

真实 RAG 测试：

```bash
RUN_LIVE_TESTS=1 uv run pytest tests/test_rag_live.py -q
```

## 目录

```text
app/
├── config.py            # 环境变量、模型/Embedding 工厂、日志
├── state.py             # AgentState
├── graph.py             # 基础图
├── entrypoint.py        # LangGraph Server 加载入口
├── cli.py               # rag / basic / mcp / multi / hitl
├── text.py              # 文本清洗
├── rag/                 # 文档加载、切分、Embedding、Chroma、RAG 图
├── mcp/                 # MCP Server / Client / Agent
├── multiagent/          # Supervisor + Knowledge/Research Agent
├── hitl/
│   ├── report.py        # 报告生成、幂等保存
│   ├── graph.py         # interrupt + Command 恢复图
│   └── cli.py           # 审批交互、暂停/恢复
└── memory/              # SQLite checkpointer 与 Store

data/
├── docs/                # 10 篇技术文档（提交）
├── chroma/              # 向量索引（不提交）
├── checkpoints.sqlite   # 会话 checkpoint（不提交）
├── memory.sqlite        # 用户偏好（不提交）
└── reports/             # 审批后保存的报告（不提交）

tests/
├── test_graph.py
├── test_text.py
├── test_rag_pipeline.py
├── test_rag_graph.py
├── test_memory.py
├── test_mcp.py
├── test_multiagent.py
├── test_hitl.py
├── test_smoke_live.py
├── test_rag_live.py
├── test_mcp_live.py
└── test_multiagent_live.py

docs/
├── day01-langgraph.md
├── day02-agentic-rag.md
├── day03-memory.md
├── day04-mcp.md
├── day05-multiagent.md
└── day06-hitl.md
```

## 开发进度

- [x] Day 1：LangGraph 骨架 + 基础 Agent
- [x] Day 2：Agentic RAG
- [x] Day 3：Memory 与持久化
- [x] Day 4：MCP
- [x] Day 5：MultiAgent
- [x] Day 6：HITL
- [ ] Day 7：Middleware
- [ ] Day 8：Runtime API 与 Web UI
- [ ] Day 9：Docker 部署
- [ ] Day 10：端到端测试与交付
