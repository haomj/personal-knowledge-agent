# Day 1：LangGraph

> 交付版本：`v0.1-basic-agent`  
> 运行环境：Python 3.12.15 / langgraph 1.2.15 / langchain 1.4.5 / langchain-core 1.6.10 / langchain-openai 1.7.1  
> 模型：DeepSeek `deepseek-flash`（OpenAI 兼容接口，`https://api.deepseek.com`）

## 1. 今日学习目标

今天要完成两件事：

1. 沿着 `StateGraph -> compile -> Pregel -> Node -> State 更新` 的调用链读 LangGraph 源码，回答五个问题：
   - Graph 如何定义执行流程？
   - State 如何在节点之间传递？
   - 节点如何决定下一步？
   - Graph 如何结束？
   - Graph 与普通 Python 函数调用有什么区别？
2. 写一个最小可运行的 Agent：`START -> agent -> END`，接收用户问题、调用 `deepseek-flash`、返回结果、支持流式输出和多轮对话，并输出基本日志。

验收标准：能够通过 Python 调用 Agent，完成多轮问答的基础执行流程。

## 2. LangGraph 核心概念

| 概念 | 一句话解释 | 在本项目中的位置 |
| --- | --- | --- |
| State | 图运行时共享的数据结构，本质是一组带 reducer 的 channel | `app/state.py` 的 `AgentState` |
| Node | 一个普通 Python 可调用对象，签名是 `State -> Partial<State>` | `app/graph.py` 的 `agent_node` |
| Edge | 节点之间的有向连接，决定下一个执行谁 | `START -> agent -> END` |
| Reducer | 同一个 state key 收到多次写入时的合并函数 | `add_messages` 把新消息追加到 `messages` |
| Compile | 校验图结构并生成可执行的 `CompiledStateGraph`（Pregel 运行时） | `builder.compile(...)` |

### State

`StateGraph` 的 State 不是普通字典，而是一份 schema 声明。每个字段会被映射成一个 channel；字段上的 `Annotated[..., reducer]` 决定这个 channel 收到多个写入时如何合并。

```python
class AgentState(MessagesState):
    ...
```

`MessagesState` 已经替我们声明了：

```python
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

所以节点返回 `{"messages": [response]}` 时，LangGraph 不会覆盖整个 `messages` 列表，而是调用 `add_messages(left, right)` 追加。

### Node

节点是普通函数，不是必须继承的类：

```python
def agent_node(state: AgentState) -> dict[str, list[AIMessage]]:
    response = model.invoke([SystemMessage(content=prompt), *state["messages"]])
    return {"messages": [response]}
```

节点只返回“增量”，不负责修改完整 State。这一点和普通函数最大的区别是：节点看到的是本 superstep 的 State 快照，写回的内容要经过 reducer 合并。

### Edge

- 普通边：`add_edge("agent", END)`，无条件进入下一个节点。
- 条件边：`add_conditional_edges("agent", router)`，由 `router(state)` 的返回值决定下一步。
- `START` / `END`：虚拟节点，表示图的入口和出口。

### Reducer

Reducer 的签名是 `(当前值, 新写入) -> 合并后的值`。`add_messages` 的默认行为是 append-only，并且会按消息 `id` 去重/替换：

- 新消息 `id` 不存在：追加。
- 新消息 `id` 已存在：替换原消息。
- `RemoveMessage`：按 `id` 删除；`REMOVE_ALL_MESSAGES` 清空。

### Compile

`StateGraph` 只是构建器，不能直接执行。`compile()` 会：

1. 校验节点和边；
2. 为每个 state 字段创建 channel；
3. 把节点包装成 `PregelNode`；
4. 返回 `CompiledStateGraph`，它继承自 `Pregel`，因此拥有 `invoke` / `stream` / `astream`。

## 3. 阅读的源码

源码路径均相对于 `.venv/lib/python3.12/site-packages/`。

| 文件 | 关键位置 | 作用 |
| --- | --- | --- |
| `langgraph/graph/state.py` | `StateGraph` 131 行；`add_node` 667 行；`add_edge` 928 行；`add_conditional_edges` 982 行；`compile` 1177 行；`CompiledStateGraph` 1404 行 | 图的声明、校验、编译 |
| `langgraph/graph/message.py` | `add_messages` 64 行；合并逻辑 205–247 行；`MessagesState` 310 行 | 消息 reducer |
| `langgraph/pregel/main.py` | `Pregel` 460 行；`stream` 2748 行；`invoke` 3924 行 | 执行引擎与流式接口 |
| `langgraph/pregel/_loop.py` | `PregelLoop` 170 行；`tick` 633 行 | 单个 superstep 的调度循环 |
| `langgraph/pregel/_algo.py` | `apply_writes` 233 行 | 把节点写入合并进 channel |
| `langgraph/pregel/_messages.py` | `StreamMessagesHandler` 58 行；`on_llm_new_token` 160 行 | `stream_mode="messages"` 的 token 捕获 |
| `langgraph/prebuilt/tool_node.py` | `ToolNode` 622 行；`tools_condition` 1582 行 | 工具执行节点与工具路由 |
| `langgraph/checkpoint/memory/__init__.py` | `InMemorySaver` 34 行 | 内存版短期记忆 |

### 3.1 StateGraph

`StateGraph` 的类文档明确写着：

> `StateGraph` is a builder class and cannot be used directly for execution. You must first call `.compile()` to create an executable graph.

也就是说它只存三样东西：

- `nodes`：节点名到 `StateNodeSpec` 的映射；
- `edges` / `waiting_edges`：普通边和多起点边；
- `branches`：条件边；
- `channels`：state 字段对应的 channel。

`add_node` 会做参数校验、推断输入 schema，并把节点包装成 `PregelNode`/`StateNodeSpec`。`add_edge` 只做合法性检查后把边加入集合；`add_conditional_edges` 把路由函数保存为 `BranchSpec`。

`compile()` 在 1177 行开始，核心步骤：

1. `ensure_valid_checkpointer`；
2. `self.validate(...)` 校验入口、出口、节点可达性；
3. 计算 `output_channels` / `stream_channels`；
4. 创建 `CompiledStateGraph`；
5. 把节点、边、分支挂到编译后的图上；
6. 返回 `compiled.validate()`（1401 行）。

`CompiledStateGraph` 继承自 `Pregel`（1404–1406 行），所以编译后的图本质上是一个 Pregel 运行时对象。

### 3.2 MessagesState

`MessagesState` 只有一行有效定义（310–311 行）：

```python
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

`add_messages` 的关键逻辑（205–247 行）：

1. 把输入统一转成 `BaseMessage`；
2. 给没有 `id` 的消息补 `uuid4`；
3. 用 `merged_by_id` 记录已有消息的位置；
4. 新 id 追加，已有 id 替换；
5. 遇到 `RemoveMessage` 时按 id 删除；
6. 遇到 `REMOVE_ALL_MESSAGES` 时只保留它之后的消息。

这解释了为什么我们第二次调用 `graph.invoke` 时，State 里的消息是 4 条：第一轮 Human + AI，第二轮 Human + AI。

### 3.3 ToolNode 与 tools_condition

Day 1 的图还没有工具，但这两个是后面 Agentic RAG、MCP、MultiAgent 的基础。

`tools_condition`（1582 行，核心逻辑 1648–1659 行）非常短：

```python
ai_message = state["messages"][-1]
if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
    return "tools"
return "__end__"
```

它只做一件事：看最后一条 AIMessage 有没有 `tool_calls`。有就路由到 `"tools"` 节点，没有就结束。

`ToolNode`（622 行）负责真正执行工具：从最后一条 AIMessage 的 `tool_calls` 中取出工具名和参数，找到对应工具，执行后把结果包装成 `ToolMessage` 写回 State。工具执行失败时会生成错误类型的 `ToolMessage`，让模型有机会自行修正。

标准 ReAct 结构因此是：

```text
START -> agent -> (有 tool_calls?) -> tools -> agent -> ... -> END
```

Day 1 只实现 `START -> agent -> END`，把工具循环留到 Day 2/4。

### 3.4 Pregel

`Pregel` 是 LangGraph 的执行引擎（`pregel/main.py:460`）。它的核心循环在 `PregelLoop`：

- `tick()`（`_loop.py:633`）准备下一个 superstep 的任务；
- `prepare_next_tasks(...)`（646 行）根据 channel 更新和边决定哪些节点要执行；
- 如果没有任务了，`self.status = "done"` 并返回 `False`（686–689 行），图结束；
- 每个节点执行后产生 writes，`apply_writes(...)`（`_algo.py:233`）把 writes 合并进 channel。

`invoke()`（`pregel/main.py:3924`）本身不是另一套执行逻辑：它在内部调用 `self.stream(..., stream_mode="values")`，只保留最后一个 State（3979 行起）。所以 `invoke` 可以理解为“跑完流，只取最终值”。

### 3.5 五个问题的答案

1. **Graph 如何定义执行流程？**  
   `StateGraph` 收集节点、边、条件边和 channel；`compile()` 校验后生成 `CompiledStateGraph`（Pregel），由 Pregel 按 superstep 调度。

2. **State 如何在节点之间传递？**  
   State 字段映射为 channel。节点读取当前 State 快照，返回部分更新；`apply_writes` 把更新写入 channel，reducer 决定如何合并。下一个节点看到的是合并后的新 State。

3. **节点如何决定下一步？**  
   三种方式：静态边 `add_edge`；条件边 `add_conditional_edges` 的路由函数；节点返回 `Command(goto=...)` 直接指定目标。

4. **Graph 如何结束？**  
   没有可执行的 task 时，`PregelLoop.tick()` 返回 `False`，状态变为 `done`。通常是因为走到了 `END`、条件边返回了结束分支，或者触发了递归/步数限制、中断。

5. **Graph 与普通 Python 函数调用有什么区别？**  
   普通函数是命令式调用栈；Graph 是声明式状态机。它额外提供：持久化 checkpoint、中断与恢复、逐 token/逐步骤流式输出、并行 superstep、reducer 合并、循环与条件路由。代价是理解成本更高，简单线性流程用普通函数更直接。

## 4. Graph 构建流程

本项目的构建代码在 `app/graph.py`：

```python
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)

compiled = builder.compile(checkpointer=checkpointer)
```

对应源码中的流程：

```text
StateGraph(AgentState)
  -> _add_schema 为 messages 建立 channel
add_node("agent", agent_node)
  -> 包装成 StateNodeSpec / PregelNode
add_edge(START, "agent")
add_edge("agent", END)
  -> 记录边
compile(checkpointer)
  -> validate() 校验结构
  -> 创建 CompiledStateGraph(Pregel)
  -> attach_node / attach_edge / attach_branch
  -> 返回 compiled.validate()
```

几个实现细节：

- `add_node` 可以根据函数签名的类型注解推断输入 schema；我们的 `agent_node(state: AgentState)` 会被识别为读取 `AgentState`。
- `compile(checkpointer=...)` 里的 checkpointer 是短期记忆：同一个 `thread_id` 的每次调用都会读写对应的 checkpoint。
- `app/entrypoint.py` 在模块导入时构建默认图，供 `langgraph.json` 使用；CLI 和测试则直接调用 `build_graph`，避免导入即创建真实模型。

## 5. Agent 工具调用流程

Day 1 的 Agent 没有工具，但先把标准流程记录下来，作为 Day 2 的实现目标：

```text
用户消息
  -> agent 节点调用 bind_tools 后的模型
  -> 模型返回 AIMessage(tool_calls=[...])
  -> tools_condition 发现 tool_calls，路由到 "tools"
  -> ToolNode 执行工具，返回 ToolMessage
  -> 回到 agent 节点
  -> 模型看到工具结果，生成最终 AIMessage
  -> tools_condition 返回 "__end__"
  -> END
```

关键点：

- `tool_calls` 是模型输出的一部分，不是 Python 直接调用工具；Agent 的“自主性”来自模型决定调用哪个工具。
- `ToolNode` 执行的是 `AIMessage.tool_calls` 中声明的工具，参数也由模型给出，因此工具实现必须自己做参数校验和错误处理。
- `tools_condition` 只看最后一条 AIMessage；如果最后一条是 `ToolMessage`，说明工具刚执行完，应该回到 agent。
- Day 1 不做工具循环，避免把基础图的 State 传递和流式问题混在一起。

## 6. State 更新机制

### 6.1 channel 与 superstep

编译时每个 state 字段变成一个 channel。一次 `invoke`/`stream` 会按 superstep 推进：

1. `prepare_next_tasks` 根据上一轮更新的 channel 找出要执行的节点；
2. 节点并行/顺序执行，产生 writes；
3. `apply_writes` 按 channel 分组写入；
4. channel 记录新版本，触发下一轮任务；
5. 没有新任务时结束。

`apply_writes` 的关键代码（`_algo.py:314–322`）：

```python
for chan, vals in pending_writes_by_channel.items():
    if chan in channels:
        if channels[chan].update(vals) and next_version is not None:
            checkpoint["channel_versions"][chan] = next_version
            if channels[chan].is_available():
                updated_channels.add(chan)
```

同一个 channel 在一个 superstep 中收到多个写入时，`channels[chan].update(vals)` 会调用 reducer 合并。

### 6.2 本项目的实测

多轮对话的日志：

```text
第一轮：agent.start messages=1
第二轮：agent.start messages=3
```

第二轮是 3 条消息：第一轮 Human、第一轮 AI、第二轮 Human。模型输入另外拼了一条 SystemMessage，但系统提示词不写入 State，所以 State 里不会每轮重复累积系统提示。

这正是我们选择“每次调用临时拼 system prompt”的原因：

- 如果把 system prompt 作为 State 字段并用 `add_messages` 追加，它会随着轮次增长；
- 如果覆盖写入，又需要额外的 reducer 规则；
- 对 Day 1 来说，临时拼接最简单，也便于后续换成 `create_agent` 的 `system_prompt` 参数。

### 6.3 checkpoint 与 thread

`compile(checkpointer=InMemorySaver())` 后，每次调用必须提供：

```python
config = {"configurable": {"thread_id": "day1-multiturn"}}
```

`thread_id` 是 checkpoint 的命名空间：

- 相同 `thread_id`：读写同一份历史，实现多轮；
- 不同 `thread_id`：彼此隔离，单元测试里已验证；
- `InMemorySaver` 只存在于进程内存，进程退出即丢失；Day 3 会换成 SQLite。

## 7. Streaming 执行流程

### 7.1 invoke 与 stream 的关系

`Pregel.invoke()` 内部调用 `self.stream(..., stream_mode="values")`，取最后一个 State。也就是说：

- `invoke` = 跑完整个流，只返回最终结果；
- `stream` = 按选择的模式逐步返回中间结果。

### 7.2 stream_mode

`stream` 的文档（`pregel/main.py:2777–2789`）列出主要模式：

| 模式 | 输出 |
| --- | --- |
| `values` | 每个 superstep 后的完整 State |
| `updates` | 每个节点返回的增量 |
| `messages` | LLM token 与 metadata 的二元组 |
| `custom` | 节点内 `StreamWriter` 写出的自定义数据 |
| `checkpoints` | checkpoint 事件 |
| `tasks` | 任务开始/结束事件 |
| `debug` | 尽可能多的调试事件 |

Day 1 的 CLI 使用 `messages` 模式，因为它能直接拿到模型 token。

### 7.3 token 如何被捕获

`StreamMessagesHandler`（`_messages.py:58`）是一个回调处理器：

- `on_chat_model_start`：记录 run_id 和 metadata；
- `on_llm_new_token`：把每个 token chunk 通过 `stream()` 回调发出去（160–173 行）；
- 节点的最终输出也会被 `_find_and_emit_messages` 收集，并用消息 id 去重。

为了让 `.invoke()` 也产生 token 回调，我们在 `get_model()` 中设置：

```python
ChatOpenAI(..., streaming=True, stream_usage=True, extra_body={"thinking": {"type": "disabled"}})
```

`streaming=True` 让模型内部走流式 HTTP；`stream_usage=True` 让流式响应也带回 token 用量。

### 7.4 CLI 的过滤逻辑

`stream_mode="messages"` 会同时产出：

- agent 节点的最终 AIMessage；
- 模型逐 token 的 AIMessageChunk；
- 其他节点的消息（未来有 tools 节点时）。

所以 CLI 只保留：

```python
metadata.get("langgraph_node") == "agent"
```

并且只打印 `AIMessage` 的文本内容。DeepSeek 关闭思考模式后 `content` 是普通字符串；如果以后开启思考模式，`reasoning_content` 在 `additional_kwargs` 里，需要单独处理。

实测“从 1 数到 10”产生了 **19 个 chunk**，说明不是伪流式（不是等完整答案再一次性返回）。

## 8. 实验结果

### 8.1 环境与配置

```dotenv
DEEPSEEK_API_KEY=***
DEEPSEEK_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-flash
LLM_TEMPERATURE=0.7
LLM_THINKING=disabled
```

`LLM_THINKING=disabled` 通过 `extra_body={"thinking": {"type": "disabled"}}` 传给 DeepSeek。实测同一句“只回答两个字：你好”只消耗 1 个 output token，说明思考模式确实关闭。

### 8.2 单元测试

命令：

```bash
uv run pytest -q
```

结果：

```text
.....ss...
8 passed, 2 skipped in 0.45s
```

覆盖内容：

- 单轮：节点能拼出 `SystemMessage + HumanMessage` 并返回答案；
- 多轮：同一 `thread_id` 第二次调用能看到第一轮历史；
- 隔离：不同 `thread_id` 不串历史；
- 流式：`stream_mode="messages"` 能拿到 agent 节点的回答；
- 输入清洗：孤立代理项会在调用模型前替换为 U+FFFD。

2 个 skipped 是真实模型冒烟测试，默认不跑。

### 8.3 真实模型冒烟测试

命令：

```bash
RUN_LIVE_TESTS=1 uv run pytest tests/test_smoke_live.py -q -s
```

结果：

```text
..
2 passed in 3.38s
```

覆盖内容：

- DeepSeek 普通调用 + 同一 thread 的第二轮能回答出第一轮告知的名字；
- DeepSeek 流式调用能拿到非空答案。

### 8.4 CLI 单轮

```bash
uv run python -m app.cli --question "用一句话介绍你自己" --thread day1-clean
```

输出（日志与答案）：

```text
19:59:42 INFO    app.graph | graph.compiled model=deepseek-flash checkpointer=InMemorySaver
19:59:42 INFO    __main__ | cli.once thread_id=day1-clean
19:59:42 INFO    app.graph | agent.start messages=1
我是你的个人知识研究助手，专注准确、简洁地回答问题，信息不足时会如实说明，绝不编造。
19:59:44 INFO    app.graph | agent.end elapsed=1.87s input_tokens=37 output_tokens=24
```

### 8.5 CLI 多轮

```bash
printf '我叫 Adrian，请记住。\n我叫什么名字？只回答名字。\n/exit\n' | \
  uv run python -m app.cli --thread day1-multiturn
```

关键输出：

```text
你> 助手> 好的，Adrian，我记住了。
你> 助手> Adrian
```

日志：

```text
agent.start messages=1
agent.end elapsed=1.47s input_tokens=39 output_tokens=8

agent.start messages=3
agent.end elapsed=1.31s input_tokens=59 output_tokens=2
```

第二轮 `messages=3`、`input_tokens` 增加到 59，说明 checkpointer 正确恢复了第一轮历史。

### 8.6 入口与打包

```bash
uv run python -c "from app.entrypoint import graph; print(type(graph).__name__)"
# CompiledStateGraph

uv run personal-knowledge-agent --help
# 正常输出 CLI 参数
```

`pyproject.toml` 中配置了：

```toml
[project.scripts]
personal-knowledge-agent = "app.cli:main"

[tool.uv.build-backend]
module-name = "app"
module-root = "."
```

这样 `uv sync` 会把项目根目录下的 `app/` 作为包安装，控制台脚本也能直接运行。

### 8.7 验收清单

| 要求 | 结果 |
| --- | --- |
| Agent 接收用户问题 | 通过（CLI / Python 均可） |
| 调用 DeepSeek 模型 | 通过（deepseek-flash） |
| 返回结果 | 通过 |
| 支持 Streaming | 通过（实测 19 个 token chunk） |
| 输出基本日志 | 通过（模型、耗时、token 数） |
| 多轮问答基础流程 | 通过（同一 thread 第二轮看到第一轮历史） |

## 9. 遇到的问题

1. **`load_dotenv()` 在 heredoc 中报 `AssertionError`**  
   在 `python - <<'EOF'` 里调用无参数的 `load_dotenv()`，`find_dotenv()` 通过调用栈找文件时失败。临时脚本改成 `load_dotenv(".env")` 即可；项目模块里正常导入时不受影响。

2. **`.env` 已被 `git add` 进暂存区**  
   虽然 `.gitignore` 写了 `.env`，但它已经在 index 里，第一次提交会把 API Key 带进历史。已执行 `git rm --cached .env`，文件仍保留在本地，只是不再被跟踪。

3. **DeepSeek 默认开启思考模式**  
   直接调用也能工作，但 Day 1 的基础问答不需要推理过程，等待时间更长、输出 token 更多。通过 `extra_body={"thinking": {"type": "disabled"}}` 关闭，并做成 `LLM_THINKING` 配置项。

4. **`stream_mode="messages"` 会重复/混合消息**  
   它既会发出节点最终的 AIMessage，也会发出逐 token 的 AIMessageChunk，未来还会有 tools 节点的消息。CLI 通过 `metadata["langgraph_node"] == "agent"` 和消息类型过滤，避免打印重复内容。

5. **`uv_build` 默认只认 `src/` 布局**  
   项目按计划使用根目录 `app/`，需要在 `pyproject.toml` 设置 `module-root = "."` 和 `module-name = "app"`，否则 `uv sync` 构建失败。

6. **第三方库日志干扰流式输出**  
   `httpx` 的每次 HTTP 请求、`langchain_openai` 的代理探测都会打 INFO 日志。`configure_logging()` 中把这些 logger 降到 WARNING，只保留 `app.*` 的关键日志。

7. **交互式输入偶发 `UnicodeEncodeError: surrogates not allowed`**  
   终端/输入法可能把中文标点（例如“，”的 UTF-8 字节 `EF BC 8C`）截断成半截字节。Python 的 stdin 默认使用 `surrogateescape`，会解出 `\udcef` 这样的孤立代理项；OpenAI SDK 用 UTF-8 编码 JSON 请求时直接崩溃，整个 CLI 会话随之中断。修复分两层：

   - `app/cli.py` 的 `_configure_stdio()` 把 stdin 的 `errors` 从 `surrogateescape` 改为 `replace`，坏字节变成 U+FFFD；
   - `app/text.py` 的 `clean_text()` / `clean_content()` 在 CLI 输入和 `agent_node` 调用模型前兜底清洗，覆盖 stdin 无法 reconfigure 的场景。

   验证方式：

   ```bash
   printf '\xef\n/exit\n' | uv run python -m app.cli --thread day1-badbyte
   ```

   修复前会在 JSON 编码阶段抛异常；修复后模型正常回复，不再中断会话。

## 10. 尚未理解的知识点

1. **Pregel 的 channel version 与 checkpoint 版本机制**  
   目前只知道 `apply_writes` 会更新 `channel_versions`，但版本号如何参与恢复、重放和 `durability` 的 `sync`/`async`/`exit` 差异还没有完整读透。

2. **`stream_mode="messages"` 在工具调用和子图下的行为**  
   Day 1 只有单节点、无工具。带 ToolNode、子图、并行节点时，token 的 namespace 和去重规则需要继续验证。

3. **ToolNode 的错误处理与并行工具调用**  
   已知失败会生成错误 `ToolMessage`，但多个 `tool_calls` 并发执行、超时、重试策略的具体实现还没读。

4. **checkpoint 的完整生命周期**  
   `get_tuple()` / `put()` / `put_writes()` 的调用时机，以及中断恢复时为什么必须复用同一个 `thread_id`，需要等 Day 3、Day 6 再深入。

5. **递归限制与循环终止**  
   图里出现环时，`recursion_limit` 如何计数、超限后 State 处于什么状态，还没有实验。

6. **DeepSeek 思考模式下的流式结构**  
   当前关闭了 thinking。开启后 `reasoning_content` 和最终 `content` 的 chunk 顺序、如何在 CLI 中区分展示，还没有实测。

## 今日交付

- 代码：`app/config.py`、`app/state.py`、`app/graph.py`、`app/cli.py`、`app/entrypoint.py`、`app/text.py`
- 测试：`tests/test_graph.py`、`tests/test_smoke_live.py`、`tests/test_text.py`
- 配置：`langgraph.json`、`pyproject.toml`、`README.md`
- 验收：单元测试 8 passed；真实模型冒烟 2 passed；CLI 单轮/多轮/流式/坏字节输入均通过
- 交付标识：`v0.1-basic-agent`（未创建 git tag；按约定不主动提交）
