# Day 8：Runtime API 与 Web UI

> 交付版本：`v0.8-web-runtime`  
> 运行环境：Python 3.12.15 / FastAPI 0.162.2 / Uvicorn 0.54.0 / LangGraph 1.2.15  
> 启动：`uv run personal-knowledge-web`  
> 地址：`http://127.0.0.1:8000`

## 1. 今日学习目标

今天要让 Agent 从 Python 程序变成浏览器可访问的应用：

1. 理解 Runtime 的基本链路：Assistant -> Thread -> Run -> Graph -> Checkpoint。
2. 提供 HTTP API：聊天、流式输出、线程状态、HITL 审批。
3. 在浏览器中完成一次 RAG 对话，看到检索/评分/生成步骤和来源。
4. 在浏览器中完成一次 HITL 审批，通过后保存报告。
5. 理解多用户/多 thread 隔离，以及本地开发服务的边界。

验收标准：

> 通过浏览器能与 Agent 对话，并完成至少一次工具调用和人工审批。

## 2. Runtime 核心概念

| 概念 | 含义 | 本项目实现 |
| --- | --- | --- |
| Assistant | 一个可调用的 Agent/图 | RAG 图、HITL 图 |
| Thread | 一次会话，对应 checkpoint key | `mode:user_id:thread_id` |
| Run | 一次执行 | 每次 POST 请求 |
| Checkpoint | 执行后的状态快照 | `SqliteSaver` |
| Streaming | 边执行边返回 | SSE `text/event-stream` |
| HITL | 暂停等待审批 | `interrupt()` + `Command(resume=...)` |

### 2.1 请求链路

```text
Browser
  -> FastAPI
  -> LangGraph (RAG / HITL)
  -> SQLite checkpoint + Store
  -> SSE / JSON 返回
```

### 2.2 Thread 与用户隔离

API 内部使用：

```python
thread_key = f"{mode}:{user_id}:{thread_id}"
```

这样：

- 同一个用户、同一个 thread：恢复历史；
- 同一个 thread 名、不同用户：互相隔离；
- RAG 和 HITL 使用不同的前缀，避免两个图的 checkpoint 混用。

## 3. 阅读的源码与关键 API

| 位置 | 作用 |
| --- | --- |
| `app/web/server.py:141` | `create_app()`，FastAPI 应用工厂 |
| `app/web/server.py:227` | RAG SSE 流式接口 |
| `app/web/server.py:301` | HITL 开始接口 |
| `app/web/server.py:337` | HITL 审批恢复接口 |
| `app/web/server.py:193` | 线程状态接口 |
| `app/web/static/index.html` | 最小 Web Chat UI |
| `langgraph/checkpoint/sqlite/__init__.py` | `SqliteSaver` |
| `langgraph/types.py:903` | `interrupt()` |
| `langgraph/types.py:833` | `Command(resume=...)` |

### 3.1 FastAPI 生命周期

[app/web/server.py:141](app/web/server.py:141) 的 `create_app()`：

```python
@asynccontextmanager
async def lifespan(app):
    with open_store(...) as store, open_checkpointer(...) as checkpointer:
        app.state.rag_graph = build_rag_graph(
            checkpointer=checkpointer, store=store
        )
        app.state.hitl_graph = build_hitl_graph(
            checkpointer=checkpointer
        )
        yield
```

**时机**：服务启动时打开 SQLite、构建图；服务停止时关闭连接。所有请求复用同一批图对象。

### 3.2 SSE

RAG 接口返回：

```python
StreamingResponse(
    event_stream(),
    media_type="text/event-stream",
)
```

每条事件：

```text
data: {"type":"token","text":"..."}

data: {"type":"step","node":"retrieve","label":"检索知识库：4 条资料"}

data: {"type":"sources","sources":[...]}

data: {"type":"done"}
```

## 4. API 设计

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| GET | `/` | Web Chat UI |
| GET | `/api/health` | 健康检查 |
| POST | `/api/threads` | 生成 thread_id |
| GET | `/api/threads/{thread_id}` | 查看消息、来源、待审批 |
| POST | `/api/chat/stream` | RAG 流式对话（SSE） |
| POST | `/api/hitl/start` | 生成报告并暂停审批 |
| POST | `/api/hitl/resume` | 审批通过/拒绝并恢复 |

### 4.1 RAG 流式接口

[app/web/server.py:227](app/web/server.py:227)：

```python
for item in graph.stream(
    state,
    config=config,
    stream_mode=["messages", "updates"],
):
    stream_mode, data = item
    if stream_mode == "messages":
        # 只把 generate 节点的 token 发给浏览器
    elif stream_mode == "updates":
        # retrieve / grade / rewrite / generate 的步骤摘要
```

为什么只发 `generate` 节点的 token？

- `grade` 和 `rewrite` 也会调用模型；
- 浏览器只需要最终回答；
- 步骤通过 `updates` 事件单独展示。

### 4.2 HITL start

[app/web/server.py:301](app/web/server.py:301)：

1. 检查该 thread 是否已有 pending interrupt；
2. 没有则 `graph.invoke(initial_hitl_state(...))` 生成报告；
3. 如果暂停，返回：

```json
{
  "status": "interrupt",
  "interrupt": {
    "title": "...",
    "content_preview": "...",
    "content_length": 1234
  }
}
```

### 4.3 HITL resume

[app/web/server.py:337](app/web/server.py:337)：

```python
result = graph.invoke(
    Command(resume={"approved": True, "feedback": "..."}),
    config,
)
```

返回：

```json
{
  "status": "saved",
  "report_path": "data/reports/....md"
}
```

## 5. Web UI

页面在 [app/web/static/index.html](app/web/static/index.html)。

### 5.1 布局

- 左侧：User ID、模式（RAG/HITL）、Thread ID、新建/加载会话；
- 中间：聊天消息、步骤标签、来源列表；
- 底部：输入框和发送按钮；
- HITL 模式下，审批请求显示为黄色审批卡片。

### 5.2 RAG 模式

前端用 `fetch` 读取 SSE：

```javascript
const reader = res.body.getReader();
const decoder = new TextDecoder();
...
buffer.split("\n\n").forEach(...)
```

处理事件：

- `token`：追加到助手消息；
- `step`：显示检索/评分/改写/生成步骤；
- `sources`：显示来源和分数；
- `error`：显示错误。

### 5.3 HITL 模式

发送后调用 `/api/hitl/start`：

- 如果返回 `interrupt`，显示审批卡片；
- 点击“通过并保存”或“拒绝”，调用 `/api/hitl/resume`；
- 保存成功显示报告路径。

### 5.4 安全

- 所有用户内容用 `textContent` 写入，不使用 `innerHTML`；
- 页面只绑定在 `127.0.0.1`；
- 没有认证，不能直接暴露到公网。

## 6. 实验结果

### 6.1 Web API 测试

```bash
uv run pytest tests/test_web.py -q
```

结果：

```text
4 passed in 0.59s
```

覆盖：

- 健康检查和静态页面；
- RAG SSE 事件（token/step/sources/done）；
- HITL start -> interrupt -> resume -> saved；
- 线程状态接口；
- 没有 pending 审批时 resume 返回 409。

### 6.2 完整离线测试

```bash
uv run pytest -q
```

结果：

```text
40 passed, 6 skipped in 3.44s
```

### 6.3 浏览器实测：RAG

启动：

```bash
uv run personal-knowledge-web
```

浏览器打开 `http://127.0.0.1:8000`，选择 Agentic RAG，提问：

```text
Agentic RAG 最多重试几次？
```

页面实际显示：

```text
Agentic RAG 的常见上限是 1 次改写、2 次检索（即最多重试 1 次）。
[来源: data/docs/03-agentic-rag.md]

检索知识库：4 条资料
资料足够
生成带来源的回答

检索来源：
- data/docs/03-agentic-rag.md (score=0.727)
- data/docs/08-deepseek-api.md (score=0.558)
- data/docs/04-mcp.md (score=0.541)
```

### 6.4 浏览器实测：HITL

切换到“HITL 报告审批”，输入：

```text
总结 Agentic RAG 的重试策略，生成一份简短报告
```

页面显示：

```text
工具调用：save_research_report（等待人工审批）
标题：Agentic RAG 的重试策略总结
```

审批卡片包含正文预览和“通过并保存/拒绝”按钮。点击“通过并保存”后：

```text
报告已保存：data/reports/Agentic-RAG-的重试策略总结-7a8725e6.md
```

### 6.5 验收清单

| 要求 | 结果 |
| --- | --- |
| 浏览器访问 | 通过 |
| RAG 流式回答 | 通过 |
| 展示检索/评分/生成步骤 | 通过 |
| 展示来源 | 通过 |
| 工具调用展示 | 通过（save_research_report 审批卡） |
| 人工审批 | 通过 |
| 审批后保存报告 | 通过 |
| 线程状态查询 | 通过 |
| 用户/线程隔离 | 通过（thread key 含 user_id） |

## 7. 遇到的问题

1. **同步图 + FastAPI 异步框架**  
   RAG/HITL 图使用同步 `SqliteSaver`，所以接口使用同步 `def`，由 FastAPI 在线程池执行；SSE 生成器也是同步的。没有把同步图硬塞进 async 端点。

2. **SSE 只发送需要的 token**  
   RAG 的 grade/rewrite 也会调用模型。如果不过滤，浏览器会看到评分和改写过程的 token。通过 `metadata["langgraph_node"] == "generate"` 只发送最终回答。

3. **HITL 重复审批**  
   `hitl_start` 先检查 pending interrupt；如果存在，直接返回，不重新生成草稿。`hitl_resume` 没有 pending 时返回 409，避免重复执行。

4. **用户与 thread 隔离**  
   直接把用户输入的 thread_id 当 checkpoint key 会让不同用户串线。API 内部使用 `mode:user_id:thread_id` 组合键。

5. **RAG 与 HITL 共用 SQLite 文件**  
   两个图共用 `data/checkpoints.sqlite`，但 thread key 加了 `rag:`/`hitl:` 前缀，避免状态冲突。

6. **UI 窄窗口布局**  
   初始断点 780px 会在预览窗口隐藏侧边栏；调整为 560px，并缩小侧边栏和聊天内边距。

7. **开发服务器不是生产服务**  
   `uvicorn` 只绑定 `127.0.0.1`，没有认证、HTTPS、CSRF、限流；不能直接暴露公网。

8. **官方 LangGraph Server / Agent Chat UI 没有采用**  
   计划允许使用官方 UI，但为了 10 天进度和可控性，选择了 FastAPI + 静态 HTML 的最小实现；后续可以替换。

## 8. 尚未理解/未实现

1. **身份认证与授权**  
   当前 User ID 由前端填写，没有登录态；生产环境需要认证，并让服务端决定用户身份。

2. **HTTPS 与反向代理**  
   只监听本机，没有 TLS、域名、反向代理和 CORS 配置。

3. **LangGraph Server / Agent Chat UI**  
   还没有使用官方 Runtime 服务、Threads API、Run API 和官方聊天前端。

4. **并发与流式取消**  
   多个请求同时写 SQLite、浏览器中断请求、服务端取消生成，还没有系统测试。

5. **审批权限**  
   任何能访问页面的人都可以审批；没有审批人身份、权限和审计日志。

6. **报告下载**  
   目前只显示保存路径，没有提供下载接口或文件预览。

7. **可观测性**  
   没有 request id、结构化日志、耗时指标或 LangSmith 追踪。

8. **部署**  
   Day 9 才会做 Docker、持久化 volume、健康检查和生产配置。

## 今日交付

- 代码：
  - `app/web/server.py`：FastAPI Runtime、SSE、HITL API、线程状态
  - `app/web/static/index.html`：最小 Web Chat UI
  - `pyproject.toml`：`personal-knowledge-web` 启动脚本
  - `.claude/launch.json`：本地预览配置
- 测试：`tests/test_web.py`（4 个）
- 验收：Web API 4 passed；完整离线 40 passed / 6 skipped；浏览器实测 RAG 流式回答、步骤/来源展示、HITL 审批保存
- 交付标识：`v0.8-web-runtime`（未创建 git tag；按约定不主动提交）
