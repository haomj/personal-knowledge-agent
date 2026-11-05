# HITL：Human-in-the-Loop 中断与恢复

HITL 让 Agent 在执行高风险操作前暂停，等待人工审批。LangGraph 提供 `interrupt()` 和 `Command(resume=...)` 来实现。

## 基本流程

```text
Agent 生成待审批内容
  -> interrupt(payload)
  -> 图暂停，checkpoint 保存状态
  -> 用户审批
  -> Command(resume=decision)
  -> 从暂停点恢复
```

## 为什么需要 Checkpointer

中断时进程可能退出，State 必须持久化。恢复时通过同一个 `thread_id` 读取 checkpoint，LangGraph 才知道停在哪个节点、下一步做什么。

没有 checkpointer，`interrupt()` 无法可靠恢复。

## 节点会重新执行

恢复时，包含 `interrupt()` 的节点会从节点开头重新执行。因此：

- `interrupt()` 之前的副作用必须幂等，或者放到恢复之后；
- 不要在中断前执行不可重复的写操作；
- 需要保存已经完成的工作，避免重复执行。

## 恢复参数

```python
graph.invoke(Command(resume={"approved": True}), config)
```

`resume` 的值会作为 `interrupt()` 的返回值。业务代码根据这个值决定保存、取消或修改。

## 测试场景

1. 审批通过，工具执行。
2. 审批拒绝，流程取消。
3. 进程重启后，用同一个 `thread_id` 恢复。
4. 重复提交审批，验证幂等性。
5. 工具执行失败，验证错误处理。

## 常见风险

- 用 `input()` 或 API 标志位模拟中断，而不是真正的 LangGraph Interrupt：无法可靠恢复。
- 恢复时换了 `thread_id`：找不到原来的 checkpoint。
- 中断前写文件或发消息：恢复时重复执行，产生重复副作用。
