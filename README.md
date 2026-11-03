# Personal Knowledge Research Agent

按 10 天计划逐步实现一个可部署的个人知识研究助手。当前进度：**Day 1 — v0.1-basic-agent**。

## 当前能力

- LangGraph 最小图：`START -> agent -> END`
- 接入 DeepSeek `deepseek-flash`（OpenAI 兼容接口）
- 命令行多轮对话（同一 `thread_id` 内保留上下文）
- 逐 token 流式输出
- 模型调用日志（耗时、token 用量）

## 环境准备

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync
```

在项目根目录创建 `.env`：

```dotenv
DEEPSEEK_API_KEY=your-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-flash
LLM_TEMPERATURE=0.7
LLM_THINKING=disabled
```

## 运行

交互式对话：

```bash
uv run python -m app.cli
```

单轮提问：

```bash
uv run python -m app.cli --question "你好，请用一句话介绍你自己"
```

同一会话内多轮：

```bash
printf '我叫 Adrian\n我叫什么名字？\n/exit\n' | uv run python -m app.cli --thread day1-demo
```

## 测试

单元测试（不访问网络）：

```bash
uv run pytest
```

真实模型冒烟测试：

```bash
RUN_LIVE_TESTS=1 uv run pytest tests/test_smoke_live.py -q
```

## 目录

```text
app/
├── config.py      # 环境变量、模型工厂、日志配置
├── state.py       # AgentState（MessagesState + add_messages）
├── graph.py       # START -> agent -> END
├── entrypoint.py  # LangGraph Server 加载入口
└── cli.py         # 命令行多轮 + 流式输出
tests/
├── test_graph.py       # 假模型单元测试
└── test_smoke_live.py  # 真实模型冒烟测试
docs/
└── day01-langgraph.md  # Day 1 学习与实验记录
```

## 开发进度

- [x] Day 1：LangGraph 骨架 + 基础 Agent
- [ ] Day 2：Agentic RAG
- [ ] Day 3：Memory 与持久化
- [ ] Day 4：MCP
- [ ] Day 5：MultiAgent
- [ ] Day 6：HITL
- [ ] Day 7：Middleware
- [ ] Day 8：Runtime API 与 Web UI
- [ ] Day 9：Docker 部署
- [ ] Day 10：端到端测试与交付
