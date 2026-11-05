# LangGraph 核心：StateGraph、compile 与 Pregel

LangGraph 用图来描述 Agent 的执行流程。核心构建器是 `StateGraph`，它本身不能直接执行，只负责收集节点、边和状态定义。

## State

State 是一份共享数据，字段通常用 `TypedDict` 声明。每个字段会被映射成一个 channel。字段可以带 reducer，例如：

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

`add_messages` 会把节点返回的新消息追加到历史里，而不是覆盖整个列表。

## Node 与 Edge

节点是普通 Python 函数，签名是 `State -> Partial<State>`。节点只返回增量，不直接修改完整 State。

边分为：

- 普通边：`add_edge("a", "b")`，无条件进入下一个节点。
- 条件边：`add_conditional_edges("a", router)`，由 `router(state)` 的返回值决定下一个节点。
- `START` 和 `END` 是虚拟节点，表示入口和出口。

## compile

`StateGraph.compile()` 会校验节点和边、创建 channel、把节点包装成 `PregelNode`，最后返回 `CompiledStateGraph`。`CompiledStateGraph` 继承自 `Pregel`，所以编译后的图有 `invoke`、`stream`、`astream` 等方法。

## Pregel 执行模型

Pregel 按 superstep 推进：

1. `prepare_next_tasks` 根据 channel 更新决定哪些节点要执行；
2. 节点执行并产生 writes；
3. `apply_writes` 把 writes 合并进 channel；
4. 没有新任务时，状态变为 `done`，图结束。

`invoke()` 本质上是运行 `stream(..., stream_mode="values")` 并只保留最后一个状态。

## 流式输出

`stream_mode="messages"` 会通过回调捕获模型 token。`StreamMessagesHandler.on_llm_new_token()` 负责把 token chunk 发出去。模型需要开启 `streaming=True`，否则 `.invoke()` 可能只产生一个完整消息，而不是逐 token。

## 常见误区

- 认为 `StateGraph` 可以直接执行：必须先 `compile()`。
- 在节点里直接修改 `state`：应该返回增量字典。
- 把系统提示词反复写进 State：会造成历史膨胀，通常在调用模型时临时拼接。
