# Docker 部署与生产工程

单实例 Agent 可以用 Docker Compose 部署。目标不是追求复杂架构，而是保证重启后仍能使用、配置不泄露、问题可定位。

## 推荐结构

```text
Docker Compose
├── Agent API
├── Web UI
├── MCP Server（可选）
└── Persistent Storage
```

stdio MCP Server 通常由 Client 作为子进程启动，不一定需要独立容器。如果要部署独立 MCP 服务，可以使用 Streamable HTTP。

## 持久化

需要持久化的数据：

- SQLite checkpoint 数据库；
- Chroma 向量索引；
- 上传或导入的文档；
- 用户长期记忆。

把这些路径挂载到 volume。容器重启后，数据库和索引仍然存在，历史会话才能恢复。

## 环境变量

API Key、模型名、数据库路径通过环境变量传入：

```text
DEEPSEEK_API_KEY=...
MODEL_NAME=deepseek-flash
EMBEDDING_MODEL=text-embedding-v4
CHROMA_DIR=/app/data/chroma
```

不要把密钥写入 Dockerfile 或镜像。`.env` 只用于本地开发，部署时使用平台提供的 secret 机制。

## 健康检查

提供一个健康检查端点，检查：

- 进程是否存活；
- 数据库是否可访问；
- 向量库是否可访问；
- 模型 API 是否配置。

健康检查不应该每次调用真实模型，否则会产生费用。

## 日志与监控

至少记录：

- 请求 id；
- 用户/会话 id；
- 每个节点耗时；
- 模型 token 用量；
- 工具调用和错误；
- 检索命中的文档。

日志要能定位一次请求的完整执行过程，同时避免记录敏感内容。

## 认证与权限

公开访问前必须加身份验证、HTTPS 和访问控制。单实例也不要把开发服务器直接暴露到公网。

## 部署顺序

1. 本地 Docker 构建成功；
2. 容器内健康检查通过；
3. 重启后历史会话恢复；
4. 再加反向代理和认证；
5. 最后考虑多实例和分布式存储。
