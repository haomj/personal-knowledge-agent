# Day 6：HITL（Human-in-the-Loop）

> 交付版本：`v0.6-hitl`  
> 运行环境：Python 3.12.15 / langgraph 1.2.15 / langgraph-checkpoint-sqlite 3.1.3  
> 核心 API：`interrupt()`、`Command(resume=...)`、SQLite checkpointer  
> 保存目录：`data/reports/`（本地生成，不提交）

## 1. 今日学习目标

今天要让 Agent 在高风险操作前暂停，等待人工审批：

1. 理解 `interrupt()` 如何暂停图。
2. 理解 `Command(resume=...)` 如何恢复。
3. 理解为什么恢复必须使用同一个 `thread_id`。
4. 理解为什么包含 `interrupt()` 的节点会从头重新执行。
5. 设计并实现一个需要审批的 `save_research_report` 流程。
6. 测试通过、拒绝、重启恢复、重复审批和保存失败。

验收标准：

> Agent 在等待审批时可以暂停，之后通过同一个 `Thread` 恢复执行。

## 2. HITL 核心概念

| 概念 | 含义 |
| --- | --- |
| Interrupt | 在节点内部暂停图，把 payload 交给客户端 |
| Resume | 客户端用 `Command(resume=value)` 把审批结果送回图 |
| Checkpointer | 保存暂停时的状态；没有它，interrupt 无法恢复 |
| Thread | 恢复时必须复用同一个 `thread_id` |
| Idempotency | 重复审批/重复保存不能产生重复副作用 |
| Node Re-execution | 恢复时，包含 `interrupt()` 的节点会从开头重新执行 |

### 2.1 暂停与恢复流程

```text
draft_report
  -> approval
       interrupt(payload)
       [图暂停，checkpoint 保存状态]
       人工审批
       Command(resume={"approved": ...})
  -> save_report / cancel
  -> END
```

### 2.2 为什么节点会重新执行

LangGraph 的 `interrupt()` 是从节点内部抛出一个可恢复的异常。恢复时，运行时重新执行该节点，`interrupt()` 这次不再抛出，而是返回 resume 值。

因此：

- `interrupt()` 之前的代码会再执行一次；
- 中断前的副作用必须幂等，或者移到中断之后；
- 需要人工输入的值通过 `interrupt()` 返回值获取。

本项目的 `approval_node` 在 `interrupt()` 之前只构造 payload，没有写文件、发消息等副作用，所以重复执行是安全的。

## 3. 阅读的源码

源码路径相对于 `.venv/lib/python3.12/site-packages/`。

| 文件 | 关键位置 | 作用 |
| --- | --- | --- |
| `langgraph/types.py` | `interrupt()` 903 行 | 在节点内暂停并返回 resume 值 |
| `langgraph/types.py` | `Command` 833 行 | 恢复执行、携带 resume 值 |
| `langgraph/types.py` | `Interrupt` 584 行 | 暂停时暴露给客户端的对象 |
| `langgraph/pregel/_algo.py` | resume 处理 1284 行附近 | 运行时把 resume 值映射回中断 |

`interrupt()` 的文档明确写了：

> The graph resumes from the start of the node, **re-executing** all logic.

以及：

> To use an `interrupt`, you must enable a checkpointer.

这两点决定了本项目的设计：

- 必须用 SQLite/InMemory checkpointer；
- 审批节点的副作用要放在 `interrupt()` 之后，或保证幂等。

## 4. Graph 设计

代码在 [app/hitl/graph.py](app/hitl/graph.py)。

```text
START
  -> draft_report
  -> approval
       ├─ approved -> save_report -> END
       └─ rejected -> cancel -> END
```

### 4.1 State

`HITLState` 继承 `AgentState`，增加：

| 字段 | 作用 |
| --- | --- |
| `report_title` | 报告标题 |
| `report_content` | 报告正文 |
| `approved` | 审批结果 |
| `feedback` | 审批意见 |
| `report_path` | 保存路径 |
| `status` | drafted/approved/rejected/saved/cancelled/error |

### 4.2 draft_report

[app/hitl/graph.py](app/hitl/graph.py) 的 `draft_report_node`：

1. 取最后一条 HumanMessage。
2. 调用 `draft_fn`（默认 `draft_report()`）。
3. 把 title/content 写回 State。

默认 `draft_report()` 在 [app/hitl/report.py](app/hitl/report.py)：

- 用 Day 2 Retriever 检索本地文档；
- 用非流式 DeepSeek 生成报告；
- 要求第一行是 `# 标题`，正文带 `[来源: ...]`。

### 4.3 approval

```python
decision = interrupt({
    "type": "approval_request",
    "action": "save_research_report",
    "title": state["report_title"],
    "content_preview": state["report_content"][:1200],
    "content_length": len(state["report_content"]),
})
```

暂停后，客户端看到：

```text
=== HITL 审批请求 ===
动作：save_research_report
标题：Agentic RAG 的重试策略研究报告
正文长度：1111 字符
预览：...
=====================
```

恢复值：

```python
Command(resume={"approved": True, "feedback": "同意"})
```

### 4.4 save_report / cancel

- `approved=True`：调用 `save_research_report()` 写入 `data/reports/`。
- `approved=False`：进入 `cancel`，不写文件。
- 保存异常：返回 `status="error"`，不抛出未处理异常。

## 5. 幂等保存

[app/hitl/report.py](app/hitl/report.py) 的 `save_research_report()`：

```python
digest = sha256(f"{title}\n{content}").hexdigest()[:8]
path = reports_dir / f"{slug}-{digest}.md"
if path.exists():
    return path
path.write_text(f"# {title}\n\n{content}\n")
```

设计点：

- 文件名由标题 + 内容哈希决定；
- 同样的标题和正文重复保存，返回同一个路径；
- 不会因为重复审批产生多个重复文件；
- 报告目录 `data/reports/` 已加入 `.gitignore`。

## 6. CLI 审批流程

`--mode hitl`，审批方式由 `--hitl-decision` 控制：

| 值 | 行为 |
| --- | --- |
| `ask`（默认） | 交互输入 y/n/p |
| `approve` | 自动通过 |
| `reject` | 自动拒绝 |
| `pause` | 打印审批请求后退出，下一次同 thread 恢复 |

### 6.1 正常审批

```bash
uv run python -m app.cli --mode hitl \
  --question "总结 Agentic RAG 的重试策略" \
  --hitl-decision approve
```

实际输出：

```text
=== HITL 审批请求 ===
动作：save_research_report
标题：Agentic RAG 的重试策略研究报告
正文长度：1111 字符
...
=====================
已保存报告：data/reports/Agentic-RAG-的重试策略研究报告-d65e2b2c.md
```

### 6.2 拒绝

```bash
uv run python -m app.cli --mode hitl \
  --question "总结 MCP 的两种传输方式" \
  --hitl-decision reject
```

输出：

```text
已取消保存。
```

### 6.3 暂停后重启恢复

第一个进程：

```bash
uv run python -m app.cli --mode hitl \
  --question "总结 Memory 的短期与长期记忆" \
  --thread day6-pause \
  --hitl-decision pause
```

输出：

```text
已暂停。进程退出后，使用同一个 --thread 再次运行，会恢复到这个审批点。
```

第二个进程：

```bash
uv run python -m app.cli --mode hitl \
  --thread day6-pause \
  --hitl-decision approve
```

输出：

```text
已保存报告：data/reports/Memory-短期与长期记忆总结报告-89b4ef01.md
```

第二个进程没有重新生成草稿，而是通过 `graph.get_state(config)` 找到 pending interrupt，再用 `Command(resume=...)` 恢复。

### 6.4 恢复逻辑

[app/hitl/cli.py](app/hitl/cli.py) 的 `run_hitl_question()`：

```python
snapshot = graph.get_state(config)
pending = _pending_interrupt(snapshot)

if pending is None:
    graph.invoke(initial_hitl_state(question), config)
    snapshot = graph.get_state(config)
    pending = _pending_interrupt(snapshot)

if pending is None:
    # 流程已经结束
    ...

decision = _make_decision(payload, decision_mode)
result = graph.invoke(Command(resume=decision), config)
```

`_pending_interrupt()` 从 `snapshot.tasks[*].interrupts` 里取第一个待审批项。

## 7. 实验结果

### 7.1 HITL 测试

```bash
uv run pytest tests/test_hitl.py -q
```

结果：

```text
5 passed in 0.71s
```

覆盖：

1. 审批通过，报告落盘；
2. 审批拒绝，不产生文件；
3. SQLite 重启后恢复 pending 审批；
4. 同样标题+正文重复保存是幂等的；
5. 保存目录异常时返回 `status="error"`，不崩溃。

### 7.2 完整离线测试

```bash
uv run pytest -q
```

结果：

```text
29 passed, 6 skipped in 3.52s
```

### 7.3 验收清单

| 要求 | 结果 |
| --- | --- |
| Agent 在审批前暂停 | 通过（`interrupt()`） |
| 通过后保存报告 | 通过 |
| 拒绝后取消 | 通过 |
| 重启后恢复审批 | 通过（同一 SQLite + thread） |
| 重复审批/保存幂等 | 通过（内容哈希文件名） |
| 工具执行失败可处理 | 通过（返回 error 状态） |

## 8. 遇到的问题

1. **恢复时节点会重新执行**  
   `interrupt()` 恢复时，`approval_node` 从头执行。因此 `interrupt()` 之前不能写文件或发消息；本项目把保存动作放到独立的 `save_report` 节点。

2. **没有 checkpointer 就无法恢复**  
   `interrupt()` 依赖 checkpoint。CLI 使用 `SqliteSaver`；如果换成 InMemorySaver，重启后 pending 状态会丢失。

3. **CLI 需要先检查 pending interrupt**  
   如果进程重启后直接 `graph.invoke(initial_state)`，会开始新一轮，而不是恢复审批。`run_hitl_question()` 先 `graph.get_state()`，发现 pending 就不重新生成草稿。

4. **暂停模式与交互模式**  
   为了方便验收，增加了 `--hitl-decision pause`。它打印审批请求后退出，下一次同 thread 运行即可恢复；默认 `ask` 仍然是人工交互。

5. **报告标题解析**  
   模型不一定会严格输出 `# 标题`。`_parse_report()` 做了降级：没有标题时用问题前 60 个字符作为标题。

6. **保存失败**  
   如果把 `reports_dir` 指向一个文件，`mkdir` 会失败。`save_report_node` 捕获异常并返回 `status="error"`，Supervisor/CLI 仍然可以展示错误，而不是整个进程崩溃。

7. **重复审批**  
   报告文件名由内容哈希决定，重复保存返回同一路径；不会产生重复文件。真正要防止重复副作用时，还需要在业务层加幂等键。

8. **审批预览长度**  
   预览只截取正文前 1200 字符，避免 interrupt payload 过大；完整正文仍在 State 中，审批通过后保存完整内容。

## 9. 尚未理解/未实现

1. **多中断点**  
   一个节点内多个 `interrupt()` 的 resume 值顺序匹配；当前只有一个审批点，还没有实践多中断点。

2. **审批时修改内容**  
   当前审批只能通过/拒绝；还不能让审批人编辑标题或正文后再保存。

3. **审批元数据与审计**  
   还没有记录审批人、审批时间、审批意见的持久化审计日志。

4. **异步 HITL**  
   当前 HITL 图使用同步 `SqliteSaver`；如果要在 MCP/MultiAgent 异步链路中嵌入 HITL，需要 `AsyncSqliteSaver` 和异步恢复。

5. **把 HITL 接入工具调用**  
   当前审批的是“保存报告”节点；还没有实现 `wrap_tool_call` 级别的高风险工具审批。

6. **多用户审批**  
   还没有设计“谁有权审批”“审批任务队列”“超时自动拒绝”等机制。

7. **报告版本管理**  
   当前用内容哈希区分版本；还没有实现同一报告的多版本历史、差异对比和回滚。

## 今日交付

- 代码：
  - `app/hitl/report.py`：报告生成、幂等保存
  - `app/hitl/graph.py`：`draft_report -> approval -> save/cancel`
  - `app/hitl/cli.py`：审批交互、暂停/恢复
  - `app/cli.py`：新增 `--mode hitl`、`--hitl-decision`
  - `app/config.py`：新增 `REPORTS_DIR`
- 测试：`tests/test_hitl.py`（5 个）
- 数据：`data/reports/`（本地生成，已忽略）
- 验收：HITL 5 passed；完整离线 29 passed / 6 skipped；通过、拒绝、重启恢复、幂等、保存失败均通过
- 交付标识：`v0.6-hitl`（未创建 git tag；按约定不主动提交）
