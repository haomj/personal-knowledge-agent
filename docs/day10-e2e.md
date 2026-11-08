# Day 10：端到端测试、源码复盘与最终交付

> 交付版本：`v1.0.0-personal-knowledge-agent`  
> 最终定位：单用户、单实例、可演示、可继续扩展的个人知识研究 Agent  
> 验收结论：核心功能、离线测试、真实 API 测试和 Docker 部署均已完成

## 1. 今日目标

Day 10 不再增加新功能，只做四件事：

1. 按验收表做端到端测试。
2. 复测并发、持久化和关键边界。
3. 整理 README、架构、API、部署与测试文档。
4. 形成 v1.0 交付说明和后续改进清单。

## 2. 端到端测试矩阵

| # | 场景 | 输入 | 预期 | 结果 |
| --- | --- | --- | --- | --- |
| 1 | 基础对话 | 普通知识问题 | 正常回答 | ✅ |
| 2 | 本地 RAG | 查询本地技术文档 | 返回内容 + 来源 | ✅ |
| 3 | Agentic RAG | 模糊/知识库没有的问题 | 改写查询、有限重试、明确说明不足 | ✅ |
| 4 | 联网研究 | 知识库没有的最新信息 | 调用 Tavily | ✅ |
| 5 | MultiAgent | 同时需要本地与网络信息 | 两个子 Agent 协作并汇总 | ✅ |
| 6 | MCP | 调用文档搜索工具 | MCP 调用成功 | ✅ |
| 7 | Memory | 重启后打开历史会话 | 保留上下文 | ✅ |
| 8 | HITL | 保存正式报告 | 暂停审批、通过后保存 | ✅ |
| 9 | Middleware | 调用受限工具 | 权限拒绝/只读重试 | ✅ |
| 10 | Persistence | 重启服务 | 历史状态恢复 | ✅ |
| 11 | Runtime 并发 | 多 thread 同时请求 | 不发生状态串线 | ✅ |

## 3. 离线测试结果

```bash
uv run pytest -q
```

```text
46 passed, 6 skipped in 3.14s
```

6 个 skipped 是需要真实 API 的测试。

新增的 Day 10 测试：

- `tests/test_concurrency.py`：两个 thread 并发执行，历史不串线；
- `tests/test_concurrency.py`：`_thread_key` 包含 mode + user，隔离不同用户/模式。

关键覆盖：

| 模块 | 测试 |
| --- | --- |
| 基础图 | `tests/test_graph.py` |
| RAG 路由 | `tests/test_rag_graph.py` |
| Chroma/切分 | `tests/test_rag_pipeline.py` |
| Memory | `tests/test_memory.py` |
| MCP | `tests/test_mcp.py` |
| MultiAgent | `tests/test_multiagent.py` |
| HITL | `tests/test_hitl.py` |
| Middleware | `tests/test_middleware.py` |
| Web API/认证 | `tests/test_web.py` |
| 部署配置 | `tests/test_deployment.py` |
| 并发隔离 | `tests/test_concurrency.py` |

## 4. 真实 API 测试结果

真实测试依赖 `.env` 中的 DeepSeek、DashScope、Tavily Key。

为了隔离异步 HTTP 客户端在不同 `asyncio.run` 之间的清理问题，**按文件分别运行**：

```bash
RUN_LIVE_TESTS=1 ./scripts/verify.sh
```

结果：

```text
test_smoke_live.py       2 passed
test_rag_live.py         1 passed
test_mcp_live.py         2 passed
test_multiagent_live.py  1 passed
```

合计：**6 passed**。

覆盖：

- DeepSeek 基础对话、多轮、流式；
- DashScope Embedding + Chroma RAG；
- MCP stdio 工具发现、检索、Agent 调用；
- Supervisor 同时委派 Knowledge/Research Agent 并汇总。

## 5. 并发与持久化

### 5.1 并发

`tests/test_concurrency.py` 使用并发线程执行两个不同 `thread_id`：

```text
thread-a -> alpha
thread-b -> beta
```

验证：

- 各自回答只包含自己的输入；
- 后续追问时，历史长度各为 4；
- 没有跨 thread 内容。

Web Runtime 使用：

```text
mode:user_id:thread_id
```

作为 checkpoint key，避免不同用户或 RAG/HITL 之间串线。

### 5.2 持久化

`tests/test_memory.py` 覆盖：

- SQLite checkpointer 重启后恢复同一 thread；
- SQLite Store 重启后偏好仍在；
- 同一用户跨 thread 共享；
- 不同用户隔离。

Docker 部署使用命名 volume `pka-data`，SQLite/Chroma/报告在容器重启后保留。

## 6. Web 与 Docker 验收

### 6.1 Web

Day 8 已在浏览器中完成：

- RAG 流式回答；
- 检索/评分/生成步骤；
- 来源列表；
- HITL 报告标题、正文预览、通过/拒绝；
- 审批通过后保存报告。

Day 10 的 Web API 测试：

```bash
uv run pytest tests/test_web.py -q
```

```text
5 passed
```

### 6.2 Docker

Day 9 已由你在 Docker Desktop 完成构建和运行验证，容器状态为 `healthy`。

当前开发环境没有 Docker daemon，因此 Day 10 没有重复执行 `docker compose build/up/restart`。部署配置由 `tests/test_deployment.py` 静态验证：

```text
3 passed
```

Docker 验收记录：

- 镜像 `personal-knowledge-agent:0.9.0` 构建成功；
- `docker compose run --rm agent python -m app.rag.ingest`：10 docs / 22 chunks；
- 容器 `healthy`；
- 端口映射 `8000->8000`；
- 曾遇到 8000 端口被旧本地 Python 进程占用，导致 `127.0.0.1` 卡住、`localhost` 正常；停止旧进程并重启 compose 后解决。

## 7. 本轮发现并修复的问题

1. **并发 thread 串线风险**  
   增加并发测试和 `mode:user_id:thread_id` 组合 key，验证不同 thread 不串线。

2. **真实 live 测试跨事件循环清理失败**  
   多个 live 文件在同一 pytest 进程中运行时，MCP/HTTP 异步客户端可能在下一个 `asyncio.run` 中清理失败。  
   处理：
   - `scripts/verify.sh` 先强制离线测试，再按文件分别运行 live 测试；
   - 子 Agent 的模型/搜索调用改为线程中执行同步客户端，减少异步连接残留。

3. **Docker BuildKit 前端镜像拉取失败**  
   移除 Dockerfile 中不必要的 `# syntax=docker/dockerfile:1.7`，避免额外拉取 BuildKit 前端镜像。

4. **Docker 代理 host.docker.internal 解析失败**  
   改用宿主机局域网 IP / 代理 Allow LAN，并把 `DOCKER_HTTP_PROXY` 传给 compose。

5. **8000 端口被旧本地进程占用**  
   旧 Python 进程占 IPv4 8000，Docker 只拿到 IPv6；停止旧进程并重启 compose 解决。

6. **API Key 暴露风险**  
   `docker compose config` 会展开 `.env`；已在文档中提醒不要贴完整输出。真实 Key 需要轮换。

## 8. 最终交付物

### 代码与配置

- `app/`：Agent、RAG、MCP、MultiAgent、HITL、Middleware、Memory、Web
- `data/docs/`：10 篇技术文档
- `tests/`：46 个离线测试 + 6 个 live 测试
- `Dockerfile`、`compose.yaml`、`.dockerignore`
- `scripts/verify.sh`

### 文档

- `README.md`：项目定位、功能、架构、Quick Start、Docker、测试、API、限制
- `docs/architecture.md`：系统架构、时序、设计取舍
- `docs/api.md`：HTTP/SSE/HITL API
- `docs/day01-...` ~ `docs/day09-...`：每日源码与实验笔记
- `docs/day10-e2e.md`：本文档
- `docs/assets/rag-demo.jpg`、`docs/assets/hitl-approval.jpg`：Web UI 演示截图

### 发布工程

- `LICENSE`：MIT
- `.github/workflows/ci.yml`：离线测试 + Docker build
- README CI / License badge
- `scripts/verify.sh`：一键离线 + live 验收

### 版本

- Python 包版本：`1.0.0`
- 交付标识：`v1.0-personal-knowledge-agent`

## 9. 已知限制

1. 共享 API Token，不是多用户登录/RBAC。
2. SQLite 仅适合单实例；多实例需要 Postgres/共享存储。
3. 无生产级 HTTPS、限流、CSRF、审计。
4. 自动记忆提取未实现；偏好通过 `/remember` 显式保存。
5. 模型调用超时/重试/熔断未统一。
6. 无可视化监控/LangSmith trace。
7. 未提供报告下载和版本管理。
8. 未实现官方 LangGraph Server / Agent Chat UI。

## 10. 后续路线

- LangSmith / OpenTelemetry：trace、指标、告警。
- Postgres checkpointer/store：多实例。
- 多用户认证与 RBAC。
- RAG 评估集：Recall@K、faithfulness、引用准确率。
- 报告下载、版本管理和审计。
- 官方 Agent Chat UI 或更完整前端。
- 模型超时、重试、熔断和成本预算。

## 今日交付结论

```text
离线测试：46 passed, 6 skipped
真实 API：6 passed（按文件隔离运行）
Web API：5 passed
部署静态测试：3 passed
并发隔离：2 passed
Docker：用户已在 Docker Desktop 验证
最终版本：v1.0.0
```

项目已达到计划目标：**从源码学习到独立设计、开发并部署一个完整的 Personal Knowledge Research Agent**。
