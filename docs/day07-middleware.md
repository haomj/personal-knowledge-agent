# Day 7：Middleware

> 交付版本：`v0.7-middleware`  
> 运行环境：Python 3.12.15 / langchain 1.4.5 / langgraph 1.2.15  
> 中间件：Logging、Permission、Retry  
> 演示 Agent：`read_knowledge` / `lookup_with_retry` / `save_research_report`

## 1. 今日学习目标

今天要理解并实现 Agent 的横切逻辑：

1. 理解 `before_model`、`after_model`、`wrap_model_call`、`wrap_tool_call`。
2. 理解 Middleware 的执行顺序。
3. 实现 Logging Middleware：模型调用、工具调用、耗时、错误。
4. 实现 Permission Middleware：按用户角色拦截高风险工具。
5. 实现 Retry Middleware：只对只读工具重试，区分有副作用的操作。
6. 验证权限拒绝和异常处理。

验收标准：

> 能从日志中观察 Middleware 的执行顺序，并验证权限拒绝和异常处理。

## 2. Middleware 核心概念

| 钩子 | 时机 | 用途 |
| --- | --- | --- |
| `before_model` | 每次模型调用前 | 注入上下文、检查状态、记录日志 |
| `after_model` | 每次模型调用后 | 检查输出、更新状态 |
| `wrap_model_call` | 包裹整个模型调用 | 计时、重试、缓存、降级 |
| `wrap_tool_call` | 包裹整个工具调用 | 权限、审计、重试、异常处理 |

### 2.1 执行顺序

本项目中间件顺序：

```python
middleware=[
    LoggingMiddleware(),
    PermissionMiddleware(),
    RetryMiddleware(),
]
```

实际观察到的顺序：

```text
Logging.before_model
Logging.wrap_model_call.start
  -> 模型调用
Logging.wrap_model_call.done
Logging.after_model

Logging.wrap_tool_call.start
  -> Permission.wrap_tool_call
       -> Retry.wrap_tool_call
            -> 业务工具
  <- Retry
  <- Permission
Logging.wrap_tool_call.done
```

也就是说：

- Logging 在最外层，能看到完整的模型/工具调用；
- Permission 在 Retry 外层，权限拒绝后不会进入重试；
- Retry 最靠近业务工具，只包裹实际执行。

### 2.2 Middleware State

`AgentMiddleware` 可以通过 `state_schema` 声明自己的状态字段，并在 `before_model`/`after_model` 返回状态增量。

本项目没有引入额外 Middleware State，而是通过：

- `runtime.context`：用户 id、角色；
- 日志：观察中间件行为；
- 工具返回值：把权限错误反馈给模型。

如果要做“调用次数统计”“预算”“审计事件”，可以给中间件增加 state_schema。

### 2.3 异常传播

- 工具抛异常：Retry 只对安全工具重试；耗尽后重新抛出；
- Logging 捕获异常、记录日志，然后 **re-raise**，不吞掉异常；
- Permission 拒绝时不抛异常，而是返回 `ToolMessage(status="error")`，让 Agent 能向用户解释。

## 3. 阅读的源码

源码路径相对于 `.venv/lib/python3.12/site-packages/`。

| 文件 | 关键位置 | 作用 |
| --- | --- | --- |
| `langchain/agents/middleware/types.py` | `AgentMiddleware` 385 行；`before_model` 455 行；`after_model` 479 行；`wrap_model_call` 503 行；`wrap_tool_call` 674 行 | Middleware 基类与钩子 |
| `langchain/agents/middleware/types.py` | `ModelRequest` 88 行 | 模型调用请求 |
| `langgraph/prebuilt/tool_node.py` | `ToolCallRequest` 132 行 | 工具调用请求与 runtime context |
| `langchain/agents/middleware/tool_retry.py` | `ToolRetryMiddleware` 32 行 | 官方工具重试实现 |
| `langchain/agents/middleware/model_retry.py` | `ModelRetryMiddleware` 33 行 | 官方模型重试实现 |
| `langchain/agents/middleware/human_in_the_loop.py` | `HumanInTheLoopMiddleware` 240 行 | 官方 HITL Middleware |
| `langchain/agents/middleware/tool_call_limit.py` | `ToolCallLimitMiddleware` 141 行 | 工具调用次数限制 |

### 3.1 ToolCallRequest

`ToolCallRequest` 包含：

```python
tool_call: ToolCall
tool: BaseTool | None
state: Any
runtime: ToolRuntime
```

本项目通过：

```python
request.tool_call["name"]
request.tool_call["args"]
request.runtime.context  # UserContext
```

实现权限判断和日志。

## 4. 演示 Agent

代码在 `app/middleware/agent.py`。

```python
agent = create_agent(
    get_model(streaming=False),
    tools=[
        read_knowledge,
        save_research_report,
        lookup_with_retry,
    ],
    middleware=[
        LoggingMiddleware(),
        PermissionMiddleware(),
        RetryMiddleware(),
    ],
    context_schema=UserContext,
    checkpointer=checkpointer,
    system_prompt=SYSTEM_PROMPT,
)
```

### 4.1 工具

| 工具 | 类型 | 行为 |
| --- | --- | --- |
| `read_knowledge(query)` | 只读 | 复用 Day 2 Retriever 查 Chroma |
| `lookup_with_retry(query)` | 只读 | 模拟前两次失败、第三次成功 |
| `save_research_report(title, content)` | 写操作/高风险 | 写入 `data/reports/` |

### 4.2 UserContext

```python
@dataclass(frozen=True)
class UserContext:
    user_id: str
    role: str = "viewer"
```

CLI 通过 `--role admin|viewer` 传入；交互中可用 `/role` 切换。

## 5. Logging Middleware

代码在 `app/middleware/logging.py`。

### 5.1 模型调用

```python
def wrap_model_call(self, request, handler):
    started = time.perf_counter()
    logger.info("mw.logging.model.start messages=%d", len(request.messages))
    try:
        response = handler(request)
    except Exception:
        logger.exception("mw.logging.model.error ...")
        raise
    logger.info("mw.logging.model.done elapsed=%.3fs", ...)
    return response
```

### 5.2 工具调用

```python
def wrap_tool_call(self, request, handler):
    name = request.tool_call.get("name")
    logger.info("mw.logging.tool.start name=%s args=%s", name, ...)
    try:
        result = handler(request)
    except Exception:
        logger.exception("mw.logging.tool.error ...")
        raise
    logger.info("mw.logging.tool.done status=%s elapsed=...", result.status, ...)
    return result
```

### 5.3 实际日志

```text
mw.logging.before_model messages=1
mw.logging.model.start messages=1
mw.logging.model.done elapsed=1.857s
mw.logging.after_model messages=2
mw.logging.tool.start name=save_research_report args={'title': ..., 'content': ...}
mw.logging.tool.done name=save_research_report status=error elapsed=0.001s ...
```

## 6. Permission Middleware

代码在 `app/middleware/permission.py`。

### 6.1 规则

```python
HIGH_RISK_TOOLS = {"save_research_report"}
```

- `role == "admin"`：允许；
- 其他角色：拒绝；
- 拒绝时返回 `ToolMessage(status="error")`，而不是抛异常。

### 6.2 关键代码

```python
context = request.runtime.context
role = getattr(context, "role", "viewer")

if name in self.high_risk_tools and role != "admin":
    return ToolMessage(
        content=f"权限拒绝：{name} 仅允许 admin 调用；当前角色 {role}。",
        tool_call_id=request.tool_call["id"],
        name=name,
        status="error",
    )
return handler(request)
```

### 6.3 实际效果

viewer：

```text
mw.permission.deny user=adrian role=viewer tool=save_research_report
[Middleware] 工具 save_research_report status=error:
权限拒绝：save_research_report 是高风险工具，仅 admin 可调用...
```

Agent 最终回答：

```text
调用已真实执行，但被权限中间件拦截了；报告没有保存。
如果需要完成这个操作，可以由 admin 账号执行...
```

admin：

```text
mw.permission.allow user=adrian role=admin tool=save_research_report
hitl.save path=data/reports/middleware-allowed-53fd8fbb.md
```

## 7. Retry Middleware

代码在 `app/middleware/retry.py`。

### 7.1 只重试安全工具

```python
SAFE_TOOL_NAMES = {"read_knowledge", "lookup_with_retry"}
```

`save_research_report` 是有副作用的操作，不自动重试：

```text
mw.retry.skip name=save_research_report reason=side_effect_or_unknown
```

### 7.2 指数退避

```python
delay = 0.1
for attempt in range(1, max_attempts + 1):
    ...
    time.sleep(delay)
    delay = min(delay * 2.0, max_delay)
```

### 7.3 实际日志

```text
mw.retry.error name=lookup_with_retry attempt=1 error=临时网络错误（第 1 次） sleep=0.10s
mw.retry.error name=lookup_with_retry attempt=2 error=临时网络错误（第 2 次） sleep=0.20s
mw.retry.success name=lookup_with_retry attempt=3
```

最终工具结果：

```text
查询成功（第 3 次）: middleware order
```

### 7.4 异常与错误结果

Retry 同时处理：

- `handler` 抛异常：捕获并重试；
- `ToolMessage.status == "error"`：对安全工具重试；
- 达到最大次数：重新抛出或返回最后一次错误结果。

## 8. 实验结果

### 8.1 单元测试

```bash
uv run pytest tests/test_middleware.py -q
```

结果：

```text
7 passed in 0.68s
```

覆盖：

1. viewer 调用高风险工具被拒绝；
2. admin 调用高风险工具通过；
3. 只读工具失败两次后第三次成功；
4. 有副作用工具不重试；
5. 重试耗尽后抛出异常；
6. Logging 包裹工具调用并记录 start/done；
7. Logging 记录工具异常后重新抛出。

### 8.2 完整离线测试

```bash
uv run pytest -q
```

结果：

```text
36 passed, 6 skipped in 3.27s
```

### 8.3 验收清单

| 要求 | 结果 |
| --- | --- |
| 从日志观察 Middleware 顺序 | 通过（Logging → Permission → Retry） |
| 权限拒绝 | 通过（viewer 被拦截，无文件） |
| admin 允许 | 通过（报告保存） |
| 只读工具重试 | 通过（第 3 次成功） |
| 有副作用工具不重试 | 通过（retry.skip） |
| 异常处理 | 通过（Logging re-raise，Retry 耗尽抛出） |

## 9. 遇到的问题

1. **Middleware 顺序必须通过日志确认**  
   不同框架版本的包装顺序可能有差异。本项目实测是 Logging 外层、Permission 中间、Retry 内层；不要只凭假设。

2. **权限拒绝返回 ToolMessage，而不是抛异常**  
   如果直接抛异常，Agent 可能中断，用户看不到可解释的错误。返回 `status="error"` 的 ToolMessage，模型可以继续生成“权限不足”的回答。

3. **重试必须排除有副作用的工具**  
   `save_research_report` 如果失败后自动重试，可能产生重复文件或重复副作用。本项目用 `SAFE_TOOL_NAMES` 白名单，只重试只读工具。

4. **工具异常与 ToolMessage error 是两种失败形态**  
   工具可能抛异常，也可能返回 `ToolMessage(status="error")`。Retry 两种都要处理，否则某些失败不会触发重试。

5. **Context 需要显式传递**  
   `UserContext` 通过 `agent.invoke(..., context=UserContext(...))` 传入；如果忘记传，权限中间件会默认 viewer 并拒绝高风险工具。

6. **Middleware 目前是同步实现**  
   MCP 和 MultiAgent 链路是异步的；要接入它们，需要实现 `awrap_model_call`/`awrap_tool_call` 等异步钩子。

7. **日志量大**  
   每次模型/工具调用都会记录 start/done。生产环境需要结构化日志、采样和敏感信息脱敏。

8. **HITL Middleware 只做了阅读**  
   官方 `HumanInTheLoopMiddleware` 可以通过 `interrupt_on` 对指定工具做审批；本项目 Day 6 用自定义 HITL 图实现，尚未把 HITL 中间件接入工具调用链。

## 10. 尚未理解/未实现

1. **Middleware State**  
   还没有用 `state_schema` 在中间件里保存调用计数、预算、审计事件。

2. **异步 Middleware**  
   MCP/MultiAgent 的异步链路需要 `awrap_model_call` / `awrap_tool_call`；当前只验证了同步 Agent。

3. **HITL Middleware 集成**  
   如何用 `HumanInTheLoopMiddleware(interrupt_on={"save_research_report": True})` 替代自定义审批图，还没有实验。

4. **Tool Call Limit / Model Call Limit**  
   LangChain 自带 `ToolCallLimitMiddleware`、`ModelCallLimitMiddleware`；还没有集成到 Supervisor 或 Middleware Agent。

5. **Model Retry / Fallback**  
   当前只做工具重试，没有做模型调用重试、模型降级或超时控制。

6. **策略配置化**  
   高风险工具、角色、重试白名单目前写在代码常量里；生产环境应放到配置或策略服务。

7. **可观测性**  
   还没有接入 LangSmith 或结构化日志，无法按 request id 聚合一次完整执行。

8. **异常语义**  
   哪些异常应该返回 ToolMessage、哪些应该中断、哪些应该重试，需要更细的异常分类和测试。

## 今日交付

- 代码：
  - `app/middleware/logging.py`
  - `app/middleware/permission.py`
  - `app/middleware/retry.py`
  - `app/middleware/tools.py`
  - `app/middleware/agent.py`
  - `app/cli.py`：新增 `--mode middleware`、`--role`
- 测试：`tests/test_middleware.py`（7 个）
- 验收：Middleware 7 passed；完整离线 36 passed / 6 skipped；权限拒绝、admin 允许、只读重试、副作用不重试、日志顺序均通过
- 交付标识：`v0.7-middleware`（未创建 git tag；按约定不主动提交）
