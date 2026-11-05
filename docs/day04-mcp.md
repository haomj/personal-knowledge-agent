# Day 4：MCP

> 交付版本：`v0.4-mcp`  
> 运行环境：Python 3.12.15 / mcp 1.30.0 / langchain-mcp-adapters 0.3.2  
> Transport：stdio  
> 工具：`list_documents()`、`search_documents(query, k=4)`

## 1. 今日学习目标

今天要把 Day 2 的本地知识库能力通过 MCP 暴露出来，并让 Agent 通过 MCP Client 自主调用：

1. 理解 MCP Server、Client、Session、Transport、Tool Schema、Tool Discovery。
2. 用 MCP Python SDK 写一个 stdio Server。
3. 把 Day 2 的 Chroma Retriever 封装成两个 MCP 工具，不重写检索逻辑。
4. 用原始 MCP Client 完成工具发现和工具调用。
5. 用 `langchain-mcp-adapters` 把 MCP 工具接入 Agent。
6. 验证 Agent 能通过 MCP Client 调用独立 MCP Server 完成知识库搜索。

## 2. MCP 核心概念

| 概念 | 含义 |
| --- | --- |
| MCP Server | 暴露工具、资源或提示词的一方 |
| MCP Client | 运行在 Agent 进程里，连接 Server、发现并调用工具 |
| Session | Client 与 Server 之间的一次连接会话 |
| Transport | 底层传输方式，常见 stdio 和 Streamable HTTP |
| Tool | Server 暴露的可调用能力，包含名称、描述和参数 schema |
| Tool Discovery | Client 通过 `list_tools` 获取 Server 的工具列表 |
| JSON-RPC | MCP 请求/响应的消息格式 |

### 2.1 调用链

```text
Agent
  -> MCP Client
  -> MCP Session
  -> MCP Server
  -> Tool
```

Agent 不直接 import Server 的函数。它只看到 Client 通过 Tool Discovery 拿到的工具 schema，然后由模型决定调用哪个工具、传什么参数。

### 2.2 stdio 与 Streamable HTTP

- **stdio**：Client 把 Server 作为子进程启动，通过 stdin/stdout 发送 JSON-RPC。适合本地工具，部署简单。
- **Streamable HTTP**：Server 作为独立 HTTP 服务运行，适合远程、多客户端场景，需要认证、超时和并发处理。

本日使用 stdio，因为知识库和 Agent 在同一台机器上，Server 可以直接复用本地 Chroma 索引。

### 2.3 为什么 stdio 下不能随便 print

stdio 的 stdout 是 JSON-RPC 协议通道。如果 Server 里用 `print()` 输出调试信息，会污染协议，导致 Client 解析失败。

因此：

- 业务日志必须走 stderr；
- 本项目 Server 默认 `MCP_LOG_LEVEL=WARNING`；
- 需要排查时再设置 `MCP_LOG_LEVEL=INFO`。

## 3. 阅读的源码

源码路径相对于 `.venv/lib/python3.12/site-packages/`。

| 文件 | 关键位置 | 作用 |
| --- | --- | --- |
| `mcp/server/fastmcp/server.py` | `FastMCP` 156 行；`run` 295 行；`tool` 462 行 | 高层 MCP Server API |
| `mcp/client/session.py` | `ClientSession` 113 行；`initialize` 160 行；`call_tool` 386 行；`list_tools` 530 行 | MCP 客户端会话 |
| `mcp/client/stdio/__init__.py` | `StdioServerParameters` 72 行；`stdio_client` 106 行 | stdio 传输 |
| `mcp/types.py` | `TextContent` 1026 行；`Tool` 1315 行；`CallToolResult` 1363 行 | 协议数据类型 |
| `langchain_mcp_adapters/client.py` | `MultiServerMCPClient` 48 行；`get_tools` 166 行 | 多 Server 工具加载 |
| `langchain_mcp_adapters/sessions.py` | `StdioConnection` 82 行 | stdio 连接配置 |
| `langchain_mcp_adapters/tools.py` | `load_mcp_tools` 539 行 | MCP Tool -> LangChain Tool |

### 3.1 FastMCP

`FastMCP` 把 MCP Server 的样板代码封装起来：

```python
mcp = FastMCP("personal-knowledge-base")

@mcp.tool()
def search_documents(query: str, k: int = 4) -> str:
    ...

mcp.run(transport="stdio")
```

函数名、docstring 和类型注解会生成 Tool Schema：

- 名称：`search_documents`
- 描述：docstring
- 参数：`query: str`、`k: int = 4`

### 3.2 ClientSession

`ClientSession` 负责初始化会话、列出工具、调用工具：

```python
async with stdio_client(params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        result = await session.call_tool("search_documents", {"query": "..."})
```

`call_tool` 返回 `CallToolResult`，其中 `content` 通常是 `TextContent` 列表；本项目从中提取文本并解析 JSON。

### 3.3 langchain-mcp-adapters

`MultiServerMCPClient` 把 MCP 工具转换成 LangChain `BaseTool`：

```python
client = MultiServerMCPClient(connection_config())
tools = await client.get_tools(server_name="knowledge")
```

转换后的工具可以直接传给 `create_agent`，Agent 不需要知道底层是 MCP。

## 4. MCP Server 实现

代码在 `app/mcp/server.py`。

### 4.1 两个工具

```text
list_documents() -> JSON {count, documents[]}
search_documents(query, k=4) -> JSON {query, count, results[]}
```

`search_documents` 复用 Day 2 的 `KnowledgeRetriever`：

```python
def _get_retriever(k):
    global _retriever
    if _retriever is None:
        _retriever = build_default_retriever(top_k=k)
    _retriever.k = k
    return _retriever
```

没有重新实现加载、切分、Embedding 或 Chroma 检索；MCP 只是把已有能力暴露出去。

### 4.2 返回 JSON 文本

工具返回 JSON 字符串，而不是 Python 对象：

```json
{
  "query": "MCP 传输方式",
  "count": 4,
  "results": [
    {
      "source": "data/docs/04-mcp.md",
      "title": "MCP：Model Context Protocol",
      "score": 0.6884,
      "content": "..."
    }
  ]
}
```

这样 Agent 能读到来源、分数和正文；CLI 或测试也可以直接 `json.loads`。

### 4.3 懒加载 Retriever

`_retriever` 是全局懒加载的：

- 启动 Server 或只调用 `list_documents` 时，不触发 Embedding/Chroma 初始化；
- 第一次 `search_documents` 时才创建 Retriever。

这让工具发现和文档列表可以在没有 API 调用的情况下工作。

### 4.4 日志

Server 启动：

```python
configure_logging(os.getenv("MCP_LOG_LEVEL", "WARNING"))
mcp.run(transport="stdio")
```

默认 WARNING，避免 stdout 污染和日志干扰；需要排查时设置 `MCP_LOG_LEVEL=INFO`。

## 5. MCP Client 与工具发现

代码在 `app/mcp/client.py`。

### 5.1 stdio 连接参数

```python
StdioServerParameters(
    command=sys.executable,
    args=["-m", "app.mcp.server"],
    cwd=str(PROJECT_ROOT),
    env=os.environ.copy(),
)
```

- `command` 使用当前虚拟环境的 Python，保证能 import `app`。
- `cwd` 指向项目根目录，保证 `.env`、`data/chroma` 路径正确。
- `env` 继承当前进程，子进程能读到 API Key。

### 5.2 工具发现

```python
async with open_session() as session:
    result = await session.list_tools()
    return list(result.tools)
```

实测返回：

```text
['list_documents', 'search_documents']
```

工具描述来自 Server 函数的 docstring。

### 5.3 直接调用工具

```python
result = await session.call_tool("list_documents", {})
```

`list_documents` 返回 10 篇文档，`search_documents` 返回带 `source`、`score` 和 `content` 的检索结果。

### 5.4 LangChain 工具加载

```python
client = MultiServerMCPClient(connection_config())
tools = await client.get_tools(server_name="knowledge")
```

适配器把 MCP Tool 的 schema 转换成 LangChain Tool；Agent 侧看到的就是普通工具。

## 6. Agent 集成

代码在 `app/mcp/agent.py`。

### 6.1 create_agent

```python
agent = create_agent(
    get_model(),
    tools=mcp_tools,
    system_prompt=system_prompt,
    checkpointer=checkpointer,
    store=store,
)
```

系统提示词要求：

- 本地知识库问题必须先调用 `search_documents`；
- 可以调用 `list_documents` 查看文档列表；
- 引用资料时使用 `[来源: 文件路径]`；
- 检索不足时明确说明，不要编造。

### 6.2 工具调用轨迹

Agent 执行后，CLI 会打印工具调用和工具返回的摘要：

```text
[MCP] 调用 search_documents {'query': 'MCP stdio 传输方式 适用场景'}
[MCP] 调用 list_documents {}
[MCP] search_documents 返回: {"query": ..., "count": 4, ...}
[MCP] list_documents 返回: {"count": 10, ...}
```

这让“Agent 是否真的通过 MCP 调用了工具”变得可观察，而不是只看最终答案。

### 6.3 异步 checkpointer

MCP 工具和 `ClientSession` 是异步 API，Agent 使用 `ainvoke`/`astream`。同步的 `SqliteSaver` 没有实现 `aget_tuple`，运行时会报：

```text
NotImplementedError: The SqliteSaver does not support async methods.
```

因此 MCP 模式使用 `AsyncSqliteSaver`：

```python
async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
    ...
```

RAG/basic 模式仍然使用同步 `SqliteSaver`。

### 6.4 CLI 模式

```bash
uv run python -m app.cli --mode mcp --user adrian --thread day4-mcp
```

也支持：

```bash
uv run python -m app.cli --mode mcp \
  --question "MCP 的 stdio 适合什么场景？" \
  --no-stream
```

`--no-stream` 会先打印工具调用记录，再打印最终答案；默认流式输出先显示答案 token，结束后打印工具轨迹。

## 7. 实验结果

### 7.1 工具发现

```text
TOOLS: ['list_documents', 'search_documents']
- list_documents | 列出本地知识库中的 Markdown/TXT 文档及其来源路径。
- search_documents | 在本地知识库中检索与 query 相关的文档片段，返回来源和分数。
```

### 7.2 list_documents

返回 10 篇文档，包含：

```json
{
  "source": "data/docs/01-langgraph-core.md",
  "relative_source": "01-langgraph-core.md",
  "title": "LangGraph 核心：StateGraph、compile 与 Pregel"
}
```

### 7.3 search_documents

查询：

```text
Agentic RAG 检索不足时最多重试几次
```

返回：

```text
- data/docs/03-agentic-rag.md 0.7293 Agentic RAG：检索、评分与查询改写
- data/docs/03-agentic-rag.md 0.6136 Agentic RAG：检索、评分与查询改写
```

`source`、`score`、`content` 都完整返回给 Agent。

### 7.4 Agent 自主调用

问题：

```text
MCP 的 stdio 适合什么场景？
```

实测 Agent 调用了：

```text
[MCP] 调用 search_documents {'query': 'MCP stdio 传输方式 适用场景'}
[MCP] 调用 list_documents {}
```

最终回答包含：

```text
stdio 适合本地工具、部署简单；
Client 把 Server 作为子进程启动；
Streamable HTTP 适合远程/多客户端，需要认证、超时和并发。
[来源: data/docs/04-mcp.md]
[来源: data/docs/10-docker-deployment.md]
```

### 7.5 跨进程恢复

MCP 模式使用 `AsyncSqliteSaver`。第一个进程问：

```text
MCP 有哪些传输方式？
```

第二个进程用同一 `thread_id` 问：

```text
我刚才问了你什么问题？
```

Agent 能复述前两轮问题和答案，说明 MCP 模式的异步 checkpoint 正常持久化。

### 7.6 测试

离线测试：

```bash
uv run pytest -q
```

结果：

```text
21 passed, 5 skipped in 3.69s
```

Day 4 新增 3 个离线 MCP 测试：

- 工具发现名称和描述；
- `list_documents` 返回至少 10 篇文档；
- LangChain 适配器能加载两个工具。

真实 MCP 测试：

```bash
RUN_LIVE_TESTS=1 uv run pytest tests/test_mcp_live.py -q
```

结果：

```text
2 passed in 11.41s
```

覆盖：

- 通过 MCP Client 调用 `search_documents` 并拿到真实 Embedding 检索结果；
- Agent 通过 MCP 工具完成一次完整问答。

### 7.7 验收清单

| 要求 | 结果 |
| --- | --- |
| 启动 MCP Server | 通过（stdio 子进程） |
| MCP Client 连接 | 通过 |
| 工具发现 | 通过（2 个工具） |
| 将 MCP Tools 接入 Agent | 通过（langchain-mcp-adapters） |
| Agent 自主调用知识库工具 | 通过（search_documents / list_documents） |
| 复用 Day 2 Retriever | 通过（没有重写检索） |
| 跨进程会话恢复 | 通过（AsyncSqliteSaver） |

## 8. 遇到的问题

1. **同步 checkpointer 不能配异步 Agent**  
   MCP Agent 使用 `ainvoke`/`astream`，同步 `SqliteSaver` 没有实现 `aget_tuple`，直接报 `NotImplementedError`。解决：MCP 模式改用 `AsyncSqliteSaver`，RAG/basic 保持同步 `SqliteSaver`。

2. **stdio 模式下 stdout 不能打印业务日志**  
   stdout 是 JSON-RPC 通道。Server 的业务日志必须走 stderr；默认 `MCP_LOG_LEVEL=WARNING`，避免污染协议和干扰 CLI 输出。

3. **Server 日志会与流式答案交错**  
   即使日志走 stderr，终端里仍会混在一起。降低默认日志级别后，CLI 主要显示 `[MCP]` 工具轨迹和答案。

4. **适配器每次工具调用会启动 Server 子进程**  
   从日志可以看到多次 `mcp.server starting`。对本地工具可以接受，但高频调用会有进程启动开销。后续可以用持久 `ClientSession` 或 `load_mcp_tools(session=...)` 复用连接。

5. **同步 Store 与异步 Agent**  
   本项目 MCP 模式把同步 `SqliteStore` 传给 `create_agent`，当前没有 Middleware 使用 Store，所以正常。如果以后在异步 Agent 中直接调用 Store，需要改用异步 Store 或异步适配。

6. **Agent 有时会多调用 `list_documents`**  
   模型可能先列文档再检索。系统提示词可以约束“优先 search_documents”，但不能保证 100% 不调用；工具调用次数和顺序仍是模型决策。

7. **工具返回 JSON 字符串，而不是结构化对象**  
   Agent 需要自己解析 JSON 文本。对当前场景足够；如果希望更严格的参数/返回值，可以使用 MCP 的 structured output 或 Pydantic 模型。

8. **stdio 只适合本地单机**  
   远程部署需要 Streamable HTTP、认证、TLS、超时和并发控制；stdio 的 Server 生命周期跟随 Client 子进程，不适合多客户端共享。

## 9. 尚未理解的知识点

1. **Streamable HTTP transport**  
   还没有实现独立 HTTP MCP Server，也没有研究认证、SSE/流式响应、会话管理。

2. **MCP Resources 和 Prompts**  
   本项目只用了 Tools；MCP 还支持 Resources、Prompts、Sampling、Elicitation 等能力，尚未实践。

3. **持久 ClientSession**  
   当前适配器每次工具调用可能启动新 Server 子进程；如何复用同一个 Session、管理生命周期和并发，还没有实现。

4. **MCP Tool 的结构化输出**  
   目前工具返回 JSON 字符串；MCP 的 structured content、output schema、错误类型还没有深入。

5. **多 MCP Server 路由**  
   `MultiServerMCPClient` 支持多个 Server；如何做工具命名冲突、权限隔离和路由，还没有实验。

6. **MCP 与 HITL/Middleware**  
   高风险 MCP 工具如何接入 `interrupt()` 审批、权限 Middleware 和审计，需要等 Day 6/7。

7. **JSON-RPC 协议细节**  
   只使用了 SDK 封装，没有逐行研究请求 id、错误码、初始化和能力协商。

8. **MCP 安全性**  
   本地 stdio 相对安全，但工具能力仍需限制；远程 MCP 的认证、TLS、输入校验和审计还没有实现。

## 今日交付

- 代码：
  - `app/mcp/server.py`：stdio MCP Server，复用 Day 2 Retriever
  - `app/mcp/client.py`：原始 MCP Client、工具发现、LangChain 工具加载
  - `app/mcp/agent.py`：MCP 工具调用 Agent、流式/非流式 CLI
  - `app/cli.py`：新增 `--mode mcp`
  - `app/memory/checkpointer.py`：新增 `AsyncSqliteSaver` 支持
- 测试：
  - `tests/test_mcp.py`（离线 3 个）
  - `tests/test_mcp_live.py`（真实 2 个）
- 验收：离线 21 passed / 5 skipped；真实 MCP 2 passed；工具发现、检索、Agent 自主调用、跨进程恢复均通过
- 交付标识：`v0.4-mcp`（未创建 git tag；按约定不主动提交）
