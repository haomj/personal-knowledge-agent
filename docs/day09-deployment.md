# Day 9：Docker 部署与生产工程

> 交付版本：`v0.9-deployed`  
> 镜像：Python 3.12 + uv  
> 编排：Docker Compose 单实例  
> 持久化：命名 volume `pka-data`  
> 健康检查：`/api/health`  
> 可选认证：`WEB_AUTH_TOKEN`

## 1. 今日学习目标

今天要把 Agent 从“本机运行”推进到“可复现的单实例部署”：

1. 编写非 root Dockerfile。
2. 用 Docker Compose 启动 Web Runtime。
3. 用命名 volume 持久化 checkpoint、Store、Chroma 和报告。
4. 配置健康检查、重启策略和优雅停止。
5. 增加可选 API Token 认证和请求日志。
6. 验证容器重启后历史会话仍可恢复。

验收标准：

> Docker 重启后，Agent 可以继续使用，并且历史 Thread 能够恢复。

## 2. 部署结构

```text
Docker Compose
  └── agent
        ├── FastAPI Web Runtime
        ├── LangGraph RAG / HITL
        ├── MCP stdio Server（子进程）
        ├── pka-data volume
        │     ├── checkpoints.sqlite
        │     ├── memory.sqlite
        │     ├── chroma/
        │     └── reports/
        └── ./data/docs:ro（知识库文档）
```

- Web UI 和 API 在同一个 FastAPI 进程里，不需要独立前端容器。
- MCP Server 使用 stdio，由 Client 作为子进程启动，不需要独立容器。
- `data/docs` 以只读 bind mount 挂载，方便更新知识库文档。
- 生成数据放在命名 volume，避免污染宿主机目录，也避免非 root 用户的权限问题。

## 3. Dockerfile

代码在 `Dockerfile`。

### 3.1 基础镜像与 uv

```dockerfile
FROM ghcr.io/astral-sh/uv:0.12.23 AS uv
FROM python:3.12-slim
```

- 使用多阶段构建，把 uv 二进制复制进运行镜像。
- 运行镜像只保留 Python 和项目依赖。
- `libgomp1` 是 onnxruntime/Chroma 需要的系统库。

### 3.2 依赖缓存

```dockerfile
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
```

先复制依赖声明并安装依赖，源码变化不会让依赖层失效。

然后复制源码和文档：

```dockerfile
COPY app ./app
COPY data/docs ./data/docs
COPY README.md langgraph.json ./
RUN uv sync --frozen --no-dev
```

`--frozen` 保证使用 `uv.lock`，构建可复现。

### 3.3 非 root

```dockerfile
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app
...
USER app
```

应用以非 root 用户运行。命名 volume 首次创建时会继承镜像中 `/app/data` 的权限，因此 app 用户可以写入。

### 3.4 健康检查

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"
```

使用 Python 标准库，不额外安装 curl。

### 3.5 启动

```dockerfile
CMD ["python", "-m", "app.web.server"]
```

`app/web/server.py` 的 `run()` 读取 `WEB_HOST` / `WEB_PORT`：

- 本机默认 `127.0.0.1:8000`
- Docker 中通过 `WEB_HOST=0.0.0.0` 监听容器外部端口

## 4. Docker Compose

代码在 `compose.yaml`。

```yaml
services:
  agent:
    build: .
    env_file: .env
    environment:
      WEB_HOST: 0.0.0.0
      WEB_PORT: "8000"
      DOCS_DIR: /app/data/docs
      CHROMA_DIR: /app/data/chroma
      CHECKPOINT_DB: /app/data/checkpoints.sqlite
      MEMORY_DB: /app/data/memory.sqlite
      REPORTS_DIR: /app/data/reports
    ports:
      - "8000:8000"
    volumes:
      - pka-data:/app/data
      - ./data/docs:/app/data/docs:ro
    init: true
    restart: unless-stopped
    healthcheck: ...
```

### 4.1 持久化

`pka-data` volume 保存：

- `checkpoints.sqlite`：短期记忆
- `memory.sqlite`：长期偏好
- `chroma/` 或 `milvus.db`：向量索引（由 `VECTOR_STORE` 决定）
- `reports/`：审批后生成的报告

`docker compose down` 不会删除 volume；`docker compose down -v` 会删除。

### 4.2 重启策略

- `restart: unless-stopped`：容器异常退出会自动重启。
- `init: true`：正确处理信号和僵尸进程。
- `stop_grace_period: 15s`：给 Uvicorn/图执行留出退出时间。

## 5. 首次启动

### 5.1 准备环境

```bash
cp .env.example .env
```

填入真实 Key。若需要认证：

```dotenv
WEB_AUTH_TOKEN=换成强随机字符串
```

### 5.2 构建

```bash
docker compose build
```

### 5.3 建索引

镜像和命名 volume 初始没有 Chroma 索引，需要先执行一次：

```bash
docker compose run --rm agent python -m app.rag.ingest
```

### 5.4 启动

```bash
docker compose up -d
```

访问：

```text
http://127.0.0.1:8000
```

查看日志：

```bash
docker compose logs -f agent
```

健康检查：

```bash
docker inspect --format '{{.State.Health.Status}}' $(docker compose ps -q agent)
```

## 6. API Token 认证

### 6.1 服务端

[app/web/server.py](app/web/server.py) 增加中间件：

- 如果 `WEB_AUTH_TOKEN` 非空；
- `/api/*`（除 `/api/health`）要求：
  - `X-API-Key: <token>`，或
  - `Authorization: Bearer <token>`；
- 不匹配返回 401。

`/api/health` 豁免，方便 Docker healthcheck。

### 6.2 前端

Web UI 左侧增加“API Token（可选）”输入框：

- 输入后保存到 `localStorage`；
- 所有 API 请求自动带 `X-API-Key`；
- 未启用认证时可以留空。

### 6.3 边界

这是单用户共享 Token，不是多用户身份系统：

- 没有用户登录、角色和审批权限；
- Token 泄露后所有 API 都可访问；
- 公网部署仍建议加 HTTPS、反向代理和限流。

## 7. 请求日志

`app/web/server.py` 的中间件记录：

```text
web.request method=POST path=/api/chat/stream status=200 elapsed=3.241s
```

Docker 日志：

```bash
docker compose logs -f agent
```

日志包含方法、路径、状态码和耗时；不记录 API Key、请求正文或报告内容。

## 8. 实验结果

### 8.1 静态部署测试

当前开发环境没有 Docker daemon（`docker` 命令不存在），因此无法在本机执行 `docker compose build/up`。新增了静态配置测试：

```bash
uv run pytest tests/test_deployment.py -q
```

结果：

```text
3 passed
```

覆盖：

- compose 有 `agent` 服务、`pka-data` volume、8000 端口、healthcheck、restart 策略；
- Dockerfile 使用 `uv sync --frozen --no-dev`、非 root `USER app`、HEALTHCHECK、`WEB_HOST=0.0.0.0`；
- `.dockerignore` 排除 `.env`、`.venv`、`data/chroma`、`data/*.sqlite*`、`data/reports`。

### 8.2 Web 认证测试

```bash
uv run pytest tests/test_web.py -q
```

结果：

```text
5 passed
```

覆盖：

- 无 Token 时 `/api/health` 可访问；
- 设置 Token 后 `/api/threads` 未带 Token 返回 401；
- 带 `X-API-Key` 返回 200。

### 8.3 完整离线测试

```bash
uv run pytest -q
```

结果：

```text
44 passed, 6 skipped
```

### 8.4 Docker 验收（需在安装 Docker Desktop 的机器执行）

由于当前环境没有 Docker，以下步骤需要在你的 Mac 上执行：

1. `docker compose build`
2. `docker compose run --rm agent python -m app.rag.ingest`
3. `docker compose up -d`
4. 浏览器打开 `http://127.0.0.1:8000`
5. 在 RAG 模式问一个问题，确认回答和来源
6. 在 HITL 模式生成报告并审批保存
7. `docker compose restart`
8. 同一 User ID + Thread ID 加载会话，确认历史仍在
9. `docker compose ps` 确认健康状态为 `healthy`

预期：

- 容器内 `/app/data/checkpoints.sqlite`、`memory.sqlite`、`chroma/`、`reports/` 存在；
- 重启后 thread 历史恢复；
- 日志中能看到 `web.request` 和 LangGraph 节点日志；
- 设置 `WEB_AUTH_TOKEN` 后，未带 Token 的 `/api/*` 返回 401。

## 9. 遇到的问题与边界

1. **当前环境没有 Docker**  
   无法实际构建和重启容器，只能做配置静态测试和本地 Web 测试。Docker 验收命令已列在 README 和本文档中。

2. **BuildKit 前端镜像拉取失败**  
   首次构建曾在 `docker.io/docker/dockerfile:1.7` 上失败，报错是代理 CONNECT 隧道 `unexpected EOF`。已移除 Dockerfile 中不必要的 `# syntax` 行；如果仍失败，检查 Docker Desktop 代理、切换代理节点或开启代理客户端的 TUN 模式。

3. **`docker compose config` 会打印明文密钥**  
   `docker compose config` 会展开 `.env`，输出中包含真实 API Key。不要把完整输出贴到聊天、Issue 或公开文档；如已暴露，应立即轮换 Key。

4. **BuildKit 无法解析 `host.docker.internal`**  
   如果构建时报 `lookup host.docker.internal: no such host`，说明 Docker 构建器无法解析宿主机别名。可以把 Docker Desktop 的 HTTP/HTTPS 代理改成宿主机局域网 IP，并确保代理允许 LAN 连接；同时在 `.env` 设置 `DOCKER_HTTP_PROXY=http://<局域网IP>:7897`。如果使用 Colima，则用 `host.lima.internal`。

5. **8000 端口被本地旧进程占用**  
   如果 `127.0.0.1:8000` 一直加载但 `localhost:8000` 正常，运行 `lsof -nP -iTCP:8000 -sTCP:LISTEN`。可能是之前本地运行的 uvicorn/`uv run personal-knowledge-web` 仍占用 IPv4 8000，而 Docker 只拿到了 IPv6。停止旧进程后执行 `docker compose down && docker compose up -d`。

6. **命名 volume 与文档更新**  
   `data/docs` 用只读 bind mount，更新宿主机文档后容器立即可见；Chroma 索引需要重新 ingest。

7. **首次启动没有索引**  
   镜像不包含 `data/chroma`，必须执行一次 `docker compose run --rm agent python -m app.rag.ingest`。

8. **SQLite 单实例**  
   `SqliteSaver` / `SqliteStore` 只适合单实例；多实例需要 Postgres/共享存储。

9. **认证是共享 Token**  
   不是多用户身份系统；没有用户登录、角色、审批权限和审计。

10. **没有 HTTPS / 反向代理**  
    当前只适合本机或内网；公网必须加 TLS、限流、CORS 和访问控制。

11. **模型/Embedding 超时**  
    工具层有 Retry Middleware，但模型调用没有统一的超时和自动重试；生产环境需要补。

12. **没有 LangSmith / 指标**  
    目前只有应用日志，没有 trace、指标和告警。

13. **镜像架构**  
    默认构建当前机器架构；如果部署到 amd64 服务器，需要 `docker buildx --platform linux/amd64`。

## 10. 尚未实现

1. 多用户认证与 RBAC。
2. HTTPS / 反向代理 / CSRF / 限流。
3. Postgres checkpointer/store 与多实例。
4. LangSmith 或 OpenTelemetry。
5. 模型调用超时、重试、熔断。
6. 备份/恢复脚本与数据迁移。
7. CI 中的 Docker build 和镜像扫描。
8. 报告下载接口和文件管理。
9. 生产级密钥管理（Docker secrets / 云 Secret Manager）。

## 今日交付

- 代码/配置：
  - `Dockerfile`
  - `compose.yaml`
  - `.dockerignore`
  - `app/web/server.py`：API Token 认证、请求日志、Web 配置
  - `app/config.py`：`WEB_HOST` / `WEB_PORT` / `WEB_AUTH_TOKEN`
  - `app/web/static/index.html`：API Token 输入与请求头
- 测试：
  - `tests/test_deployment.py`（3 个）
  - `tests/test_web.py`（5 个，含认证）
- 验收：
  - 静态部署测试 3 passed
  - Web 测试 5 passed
  - 完整离线 44 passed / 6 skipped
  - Docker 实际构建/重启：当前环境无 Docker，需在 Docker Desktop 环境执行
- 交付标识：`v0.9-deployed`（未创建 git tag；按约定不主动提交）
