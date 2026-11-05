# DeepSeek API 与 deepseek-flash

DeepSeek 提供 OpenAI 兼容的 Chat Completions 接口，可以直接用 `langchain_openai.ChatOpenAI` 调用。

## 基本配置

```python
ChatOpenAI(
    model="deepseek-flash",
    base_url="https://api.deepseek.com",
    api_key=os.environ["DEEPSEEK_API_KEY"],
)
```

`deepseek-flash` 是当前官方模型名，适合日常对话和工具调用。另一个模型是 `deepseek-v4-pro`。

## 思考模式

DeepSeek 默认可能开启思考模式。对于简单问答，可以通过请求参数关闭：

```python
extra_body={"thinking": {"type": "disabled"}}
```

关闭后响应更快、输出 token 更少。需要推理链时再开启。

## 工具调用

DeepSeek 支持 tool calling。模型返回的 `AIMessage` 中包含 `tool_calls`，LangChain 的 `bind_tools` 可以正常使用。

## 结构化输出的兼容性

在我们的兼容端点上，`with_structured_output()` 默认使用的 JSON schema `response_format` 会返回：

```text
This response_format type is unavailable now
```

但使用函数调用方式可以工作：

```python
model.with_structured_output(MyModel, method="function_calling")
```

因此做检索评分时，优先使用 `method="function_calling"`；如果供应商不支持，再退回普通文本解析。

## 流式输出

`ChatOpenAI(..., streaming=True)` 可以逐 token 输出。配合 LangGraph 的 `stream_mode="messages"`，可以在 CLI 或 Web UI 中实时显示。

## 超时与重试

生产环境要设置：

- 请求超时；
- 最大重试次数；
- 只对幂等操作自动重试；
- 记录请求耗时和错误码。

不要把 API Key 写入源码或提交到 git。
