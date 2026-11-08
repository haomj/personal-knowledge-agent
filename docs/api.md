# API 说明

Base URL：

```text
http://127.0.0.1:8000
```

FastAPI 自动文档：

```text
http://127.0.0.1:8000/docs
```

## 1. 认证

可选共享 Token，由 `WEB_AUTH_TOKEN` 控制。

- 未设置：本地开发不启用认证。
- 设置后：除 `/api/health` 外的 `/api/*` 需要：

```http
X-API-Key: <token>
```

或：

```http
Authorization: Bearer <token>
```

`/api/health` 始终豁免，供 Docker healthcheck 使用。

> 这是单用户共享 Token，不是多用户登录/RBAC。

## 2. 通用参数

| 参数 | 说明 |
| --- | --- |
| `user_id` | 用户标识，影响长期记忆和 thread 隔离 |
| `thread_id` | 逻辑会话 ID |
| `mode` | `rag` 或 `hitl`（线程状态查询用） |

服务端实际 checkpoint key：

```text
<mode>:<user_id>:<thread_id>
```

因此不同用户使用相同 `thread_id` 不会串线。

## 3. 健康检查

```http
GET /api/health
```

响应：

```json
{
  "status": "ok",
  "model": "deepseek-flash",
  "modes": ["rag", "hitl"],
  "auth_required": false
}
```

## 4. 新建 thread

```http
POST /api/threads
```

响应：

```json
{
  "thread_id": "a1b2c3d4e5f6"
}
```

## 5. 查询 thread 状态

```http
GET /api/threads/{thread_id}?mode=rag&user_id=web-user
```

响应：

```json
{
  "thread_id": "demo",
  "mode": "rag",
  "user_id": "web-user",
  "messages": [
    {
      "role": "user",
      "content": "Agentic RAG 最多重试几次？"
    },
    {
      "role": "assistant",
      "content": "最多 1 次改写、2 次检索……"
    }
  ],
  "sources": [
    {
      "source": "data/docs/03-agentic-rag.md",
      "score": 0.727
    }
  ],
  "status": "",
  "report_path": "",
  "interrupt": null
}
```

HITL 暂停时：

```json
{
  "status": "approved",
  "interrupt": {
    "type": "approval_request",
    "action": "save_research_report",
    "title": "Agentic RAG 的重试策略总结",
    "content_preview": "……",
    "content_length": 1234
  }
}
```

## 6. RAG 流式对话

```http
POST /api/chat/stream
Content-Type: application/json
```

请求：

```json
{
  "message": "Agentic RAG 最多重试几次？",
  "thread_id": "demo",
  "user_id": "web-user"
}
```

响应为 SSE：

```http
Content-Type: text/event-stream
```

事件序列：

```text
data: {"type":"start","mode":"rag","thread_id":"demo"}

data: {"type":"step","node":"retrieve","label":"检索知识库：4 条资料"}

data: {"type":"step","node":"grade","label":"资料足够","reason":"..."}

data: {"type":"step","node":"generate","label":"生成带来源的回答"}

data: {"type":"token","text":"Agentic RAG "}

data: {"type":"token","text":"最多重试 1 次……"}

data: {"type":"sources","sources":[{"source":"data/docs/03-agentic-rag.md","score":0.727}]}

data: {"type":"done"}
```

错误：

```text
data: {"type":"error","message":"..."}
```

### 前端读取示例

```javascript
const response = await fetch("/api/chat/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ message, thread_id, user_id }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });
  const parts = buffer.split("\n\n");
  buffer = parts.pop();
  for (const part of parts) {
    const line = part.split("\n").find((l) => l.startsWith("data: "));
    if (!line) continue;
    const event = JSON.parse(line.slice(6));
    // token / step / sources / done / error
  }
}
```

## 7. HITL 开始

```http
POST /api/hitl/start
Content-Type: application/json
```

请求：

```json
{
  "message": "总结 Agentic RAG 的重试策略",
  "thread_id": "hitl-demo",
  "user_id": "web-user"
}
```

### 第一次调用：生成报告并暂停

```json
{
  "status": "interrupt",
  "thread_id": "hitl-demo",
  "interrupt": {
    "type": "approval_request",
    "action": "save_research_report",
    "title": "Agentic RAG 的重试策略总结",
    "content_preview": "……",
    "content_length": 1234
  }
}
```

服务端行为：

1. 如果该 thread 已有 pending interrupt，直接返回该审批请求；
2. 否则生成报告并暂停；
3. 不会自动保存。

## 8. HITL 审批恢复

```http
POST /api/hitl/resume
Content-Type: application/json
```

通过：

```json
{
  "thread_id": "hitl-demo",
  "user_id": "web-user",
  "approved": true,
  "feedback": "同意保存"
}
```

响应：

```json
{
  "status": "saved",
  "report_path": "data/reports/Agentic-RAG-的重试策略总结-7a8725e6.md",
  "feedback": "同意保存"
}
```

拒绝：

```json
{
  "thread_id": "hitl-demo",
  "user_id": "web-user",
  "approved": false,
  "feedback": "标题需要修改"
}
```

响应：

```json
{
  "status": "cancelled",
  "report_path": "",
  "feedback": "标题需要修改"
}
```

没有 pending 审批：

```http
409 Conflict
```

```json
{
  "detail": "该线程没有待审批的 HITL 任务"
}
```

保存失败时：

```json
{
  "status": "error",
  "report_path": "",
  "feedback": "保存失败：..."
}
```

## 9. 错误码

| 状态码 | 场景 |
| --- | --- |
| 200 | 成功 |
| 401 | API Token 缺失或错误 |
| 409 | HITL 没有待审批任务 |
| 422 | 请求体校验失败（如空 message） |
| 500 | 未捕获服务端错误 |

SSE 流中的业务错误通过 `{"type":"error"}` 事件返回。

## 10. 请求限制

- `message`：1～8000 字符；
- `thread_id`、`user_id`：1～120 字符；
- 没有请求限流；公网部署需在反向代理层加限流；
- 没有 HTTPS；公网部署必须加 TLS。

## 11. curl 示例

健康检查：

```bash
curl http://127.0.0.1:8000/api/health
```

RAG（SSE）：

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"Agentic RAG 最多重试几次？","thread_id":"api-demo","user_id":"adrian"}'
```

HITL 开始：

```bash
curl -X POST http://127.0.0.1:8000/api/hitl/start \
  -H 'Content-Type: application/json' \
  -d '{"message":"总结 Agentic RAG 的重试策略","thread_id":"api-hitl","user_id":"adrian"}'
```

HITL 通过：

```bash
curl -X POST http://127.0.0.1:8000/api/hitl/resume \
  -H 'Content-Type: application/json' \
  -d '{"thread_id":"api-hitl","user_id":"adrian","approved":true,"feedback":"同意"}'
```

如果启用了 Token，加上：

```http
-H 'X-API-Key: <token>'
```
