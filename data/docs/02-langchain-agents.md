# LangChain 1.x 的 Agent 与 create_agent

LangChain 1.x 提供了更高层的 Agent 封装 `create_agent`。它适合快速构建 ReAct 风格的工具调用 Agent，而 LangGraph 更适合需要显式状态、循环、中断和持久化的流程。

## create_agent 的基本结构

```python
agent = create_agent(
    model,
    tools=[search_docs, web_search],
    system_prompt="你是一个研究助手",
    checkpointer=checkpointer,
    store=store,
)
```

`create_agent` 内部仍然会构建一个 LangGraph 图。模型节点负责决定是否调用工具，工具节点负责执行，条件边根据 `AIMessage.tool_calls` 决定是否回到模型。

## Tool Calling

模型通过 `bind_tools([...])` 获得工具 schema。模型输出的 `AIMessage` 里会包含 `tool_calls`，每个调用包含工具名、参数和调用 id。工具执行结果会包装成 `ToolMessage` 写回消息历史。

工具函数的 docstring 和类型注解会影响生成的 schema，因此参数要写清楚，工具职责要单一。

## 与 StateGraph 的关系

- `create_agent`：开箱即用，适合标准工具循环。
- `StateGraph`：需要自定义节点、条件路由、人工审批、检索评分时更灵活。

在同一个项目里可以混用：用 `create_agent` 构建子 Agent，再把它作为 LangGraph 的一个节点。

## Middleware

LangChain 1.x 支持 Middleware，用来插入日志、权限检查、重试和 HITL。常见钩子包括 `before_model`、`after_model`、`wrap_model_call`、`wrap_tool_call`。Middleware 适合横切逻辑，不应该替代业务节点。

## 消息类型

常见消息类型：

- `SystemMessage`：系统指令。
- `HumanMessage`：用户输入。
- `AIMessage`：模型输出，可能包含 `tool_calls`。
- `ToolMessage`：工具执行结果。

理解消息类型是理解 Agent 循环的基础。
