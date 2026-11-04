# Personal Knowledge Research Agent

按 10 天计划逐步实现一个可部署的个人知识研究助手。当前进度：**Day 2 — v0.2-agentic-rag**。

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
- 保留 Day 1 的基础对话模式

```text
用户问题
  -> retrieve（Chroma）
  -> grade（资料是否足够）
     -> 足够：generate（带来源回答）
     -> 不足：rewrite -> retrieve（最多一次）
              -> generate（资料不足时明确说明）
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
```

> `.env` 已被 `.gitignore` 忽略，不要提交真实 Key。`data/chroma/` 是生成的索引，也不要提交。

## 构建知识库索引

```bash
uv run python -m app.rag.ingest
```

或者使用安装后的命令：

```bash
uv run personal-knowledge-ingest
```

默认会加载 `data/docs/` 下的 Markdown/TXT，切分后重建 `data/chroma/`。
加 `--append` 可以不清空旧索引直接追加。

## 运行

默认使用 Agentic RAG：

```bash
uv run python -m app.cli
```

单轮提问：

```bash
uv run python -m app.cli --question "Agentic RAG 最多重试几次？"
```

回到 Day 1 的基础对话：

```bash
uv run python -m app.cli --mode basic
```

同一会话内多轮：

```bash
printf 'MCP 有哪些传输方式？\n其中 stdio 适合什么场景？\n/exit\n' | \
  uv run python -m app.cli --thread day2-multi
```

关闭流式输出：

```bash
uv run python -m app.cli --question "Chroma 推荐什么距离度量？" --no-stream
```

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
├── state.py             # AgentState（messages + RAG 字段）
├── graph.py             # Day 1 基础图
├── entrypoint.py        # LangGraph Server 加载入口
├── cli.py               # 命令行：RAG / 基础对话
├── text.py              # 文本清洗
└── rag/
    ├── loader.py        # Markdown/TXT 加载
    ├── splitter.py      # 递归切分
    ├── embeddings.py    # Embedding 工厂
    ├── vector_store.py  # Chroma 创建/重建
    ├── retriever.py     # 相似度检索
    ├── evaluator.py     # Retrieval Grading
    ├── rewriter.py      # Query Rewriting
    ├── graph.py         # Agentic RAG 图
    └── ingest.py        # 索引构建脚本

data/
├── docs/                # 10 篇本地技术文档（会提交）
└── chroma/              # 生成的向量索引（不提交）

tests/
├── test_graph.py        # Day 1 基础图
├── test_text.py         # 输入清洗
├── test_rag_pipeline.py # 加载/切分/Chroma
├── test_rag_graph.py    # RAG 路由
├── test_smoke_live.py   # Day 1 真实模型
└── test_rag_live.py     # Day 2 真实 RAG

docs/
├── day01-langgraph.md
└── day02-agentic-rag.md
```

## 开发进度

- [x] Day 1：LangGraph 骨架 + 基础 Agent
- [x] Day 2：Agentic RAG
- [ ] Day 3：Memory 与持久化
- [ ] Day 4：MCP
- [ ] Day 5：MultiAgent
- [ ] Day 6：HITL
- [ ] Day 7：Middleware
- [ ] Day 8：Runtime API 与 Web UI
- [ ] Day 9：Docker 部署
- [ ] Day 10：端到端测试与交付
