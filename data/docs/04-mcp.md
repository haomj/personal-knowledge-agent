# MCP：Model Context Protocol

MCP（Model Context Protocol）是一套让模型应用连接外部工具和数据源的协议。它把“工具提供方”和“工具使用方”解耦：工具由 MCP Server 暴露，Agent 通过 MCP Client 调用。

## 核心角色

- **MCP Server**：暴露工具、资源或提示词。例如一个文档搜索 Server 提供 `search_documents(query)`。
- **MCP Client**：运行在 Agent 进程里，连接 Server，获取工具列表并调用工具。
- **Session**：Client 与 Server 之间的一次连接会话。
- **Transport**：底层传输方式，常见的是 stdio 和 Streamable HTTP。

## 通信过程

```text
Agent
  -> MCP Client
  -> MCP Session
  -> MCP Server
  -> Tool
```

Client 先通过工具发现（Tool Discovery）获取 Server 提供的工具 schema，再把它们转换成模型能理解的工具定义。模型决定调用哪个工具后，Client 通过 JSON-RPC 发送请求，Server 执行并返回结果。

## stdio 与 Streamable HTTP

- **stdio**：Client 把 Server 作为子进程启动，通过标准输入输出通信。适合本地工具，部署简单。
- **Streamable HTTP**：Server 作为独立 HTTP 服务运行，适合远程或多客户端场景。需要处理认证、超时和并发。

## 工具 Schema

工具名、描述和参数类型会作为 schema 暴露给模型。描述要清晰，参数要尽量少而明确。Server 端仍然必须校验参数，不能假设模型一定传对。

## 与 RAG 的关系

Day 2 的 Retriever 可以直接封装成 MCP Tool，例如：

```text
search_documents(query: str) -> list[Document]
list_documents() -> list[str]
```

这样 Agent 不必重写检索逻辑，只是把已有能力通过 MCP 暴露出去。

## 安全注意

- 限制工具能力，不要暴露任意文件读写。
- 对高风险工具增加人工审批。
- 记录每次工具调用的参数和结果。
- 远程 MCP Server 必须加认证和 TLS。
