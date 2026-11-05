# Memory：短期记忆与长期记忆

Agent 的记忆通常分两层：短期记忆保存一次会话的上下文，长期记忆保存跨会话的用户偏好或事实。

## 短期记忆：Checkpointer

LangGraph 的 checkpointer 把每个 superstep 后的 State 保存下来。调用图时通过 `thread_id` 指定会话：

```python
config = {"configurable": {"thread_id": "user-123-chat-1"}}
graph.invoke(inputs, config)
```

同一个 `thread_id` 会读写同一份历史，实现多轮对话；不同 `thread_id` 彼此隔离。

常见 checkpointer：

- `InMemorySaver`：进程内存，适合测试，重启即丢。
- `SqliteSaver`：本地 SQLite，适合单实例部署。
- Postgres Saver：适合多实例和生产环境。

## Checkpoint 的内容

checkpoint 不只有消息列表，还包含：

- channel 的当前值；
- channel 版本；
- 待处理的 writes；
- 下一步要执行的节点；
- 中断信息。

因此它不仅能恢复对话，还能从中断点继续执行。

## 长期记忆：Store

LangGraph Store 用 namespace + key 保存结构化数据。namespace 通常按租户和用户隔离：

```python
namespace = ("tenant_a", "user_001")
store.put(namespace, "preferences", {"language": "zh"})
```

读取时使用同样的 namespace。不同用户使用不同 namespace，就不会串数据。

## 短期与长期的分工

- 短期记忆：这次对话说过什么，用于理解上下文。
- 长期记忆：用户长期偏好、项目背景、重要事实，跨会话共享。

第一版可以通过显式工具保存用户偏好，不必马上做自动记忆提取。自动提取容易写入错误信息，需要额外的审核和去重机制。

## 生产环境风险

SQLite checkpointer 适合单实例。多实例部署时，不同实例看不到彼此的本地数据库，需要改用共享存储，并考虑并发写入和锁。
