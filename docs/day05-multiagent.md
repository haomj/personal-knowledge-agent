# Day 5：MultiAgent

> 交付版本：`v0.5-multiagent`  
> 运行环境：Python 3.12.15 / langgraph 1.2.15 / langchain 1.4.5 / langchain-tavily / mcp 1.30.0  
> 架构：Supervisor + Knowledge Agent + Research Agent  
> 交付目标：一个问题同时需要本地文档和互联网信息时，Supervisor 能调用两个子 Agent 并汇总结果。

## 1. 今日学习目标

今天要理解并实现一个最小多 Agent 系统：

1. 理解 Supervisor、Agent Handoff、Task Delegation、Context Isolation、Result Aggregation。
2. 实现 Supervisor：判断任务类型、委派子任务、汇总结果。
3. 实现 Knowledge Agent：通过 MCP 检索本地知识库。
4. 实现 Research Agent：通过 Tavily 联网搜索。
5. 限制任务数量：最多 3 次子任务、每个 Agent 最多 2 次。
6. 验证 Supervisor 能同时调用两个 Agent 并整合结果。

验收标准：

> 用户提出一个需要同时查询本地文档和互联网的问题，Supervisor 能调用两个 Agent 并整合结果。

## 2. MultiAgent 核心概念

| 概念 | 含义 | 本项目实现 |
| --- | --- | --- |
| Supervisor | 判断任务类型、分配任务、汇总结果 | `build_multiagent()` 创建的 `create_agent` |
| Agent Handoff | 把子任务交给另一个 Agent | `ask_knowledge_agent` / `ask_research_agent` 工具 |
| Task Delegation | 把用户问题拆成子问题 | Supervisor 提示词 + 模型生成子问题 |
| Context Isolation | 子 Agent 不共享完整上下文 | 子 Agent 只收到 `question` 字符串 |
| Result Aggregation | 汇总多个子 Agent 的结果 | Supervisor 生成最终答案 |

### 2.1 为什么用 Tool-based Handoff

计划建议使用 **Supervisor + Tool-based Handoff**，而不是复杂的多层嵌套 Graph。

原因：

- Supervisor 仍然是一个普通 `create_agent`，工具调用就是 handoff；
- 子 Agent 是普通 async 函数，不需要共享 State；
- 容易加任务预算、日志和错误处理；
- 后续可以逐步替换成更复杂的 LangGraph 路由。

### 2.2 本项目的角色

```text
Supervisor
  ├─ ask_knowledge_agent(question)
  │    -> Knowledge Agent
  │    -> MCP search_documents
  │    -> Chroma
  │    -> LLM 总结
  └─ ask_research_agent(question)
       -> Research Agent
       -> Tavily
       -> LLM 总结
  -> Supervisor 汇总最终答案
```

### 2.3 Context Isolation

Knowledge Agent 和 Research Agent 的函数签名只有：

```python
async def run_knowledge_agent(query: str, ...) -> SubagentResult
async def run_research_agent(query: str, ...) -> SubagentResult
```

它们看不到 Supervisor 的完整消息历史，也看不到另一个子 Agent 的结果。Supervisor 只把“子问题”传进去，拿回“答案 + 来源”。

这样做的好处：

- 子 Agent 的 prompt 更短、更聚焦；
- 不会把无关上下文带进检索/搜索；
- 子 Agent 可以独立测试和替换。

## 3. 参考架构

计划建议参考 Open Deep Research 的 `src/legacy/multi_agent.py`，重点理解：

- Supervisor 如何分配任务；
- Agent Handoff 如何发生；
- Task Delegation 如何拆分子问题；
- Context Isolation 如何实现；
- Result Aggregation 如何汇总。

本次实现说明：

- 当前环境对 GitHub raw 的访问被限制（`HTTP 000`），没有逐行拉取归档源码；
- 因此按计划描述的 Supervisor–Researcher 架构，实现了最小可运行版本；
- 没有直接复制第三方项目结构，而是复用本项目已有的 RAG、MCP、Memory 和 CLI。

## 4. 代码结构

```text
app/multiagent/
├── __init__.py
├── models.py        # SubagentResult
├── knowledge.py     # Knowledge Agent
├── research.py      # Research Agent
└── supervisor.py    # Supervisor + TaskBudget + CLI
```

### 4.1 SubagentResult

[app/multiagent/models.py:9](app/multiagent/models.py:9)：

```python
@dataclass(frozen=True)
class SubagentResult:
    agent: str
    query: str
    answer: str
    sources: list[str]
```

`to_text()` 把结果格式化成 Supervisor 工具输出：

```text
[knowledge] 子任务：...
回答：...
来源：
- data/docs/03-agentic-rag.md
```

Supervisor 只看到这段文本，看不到子 Agent 的完整消息历史。

## 5. Knowledge Agent

[app/multiagent/knowledge.py:43](app/multiagent/knowledge.py:43)：

```python
async def run_knowledge_agent(
    query: str,
    *,
    search: SearchFn | None = None,
    model: BaseChatModel | None = None,
    top_k: int = 4,
) -> SubagentResult:
```

执行顺序：

1. `search(query, top_k)`：默认调用 [app/multiagent/knowledge.py:21](app/multiagent/knowledge.py:21) 的 `mcp_search()`。
2. `mcp_search()` 通过 Day 4 的 MCP Client 调用 `search_documents`。
3. Server 内部复用 Day 2 的 `KnowledgeRetriever` 和 Chroma。
4. 把检索结果格式化为 context。
5. 用非流式模型生成答案。
6. 收集 `source` 列表，返回 `SubagentResult`。

关键点：

- **复用 MCP**：没有重新写检索逻辑。
- **只拿子问题**：函数没有 Supervisor 的历史消息。
- **资料不足要说明**：prompt 要求“资料不足时明确说明”。
- **来源可追溯**：返回 `data/docs/...` 路径。

如果 MCP 或检索失败，返回 `answer="本地知识库检索失败：..."`，不会让 Supervisor 崩溃。

## 6. Research Agent

[app/multiagent/research.py:49](app/multiagent/research.py:49)：

```python
async def run_research_agent(
    query: str,
    *,
    search: SearchFn | None = None,
    model: BaseChatModel | None = None,
    max_results: int = 5,
) -> SubagentResult:
```

执行顺序：

1. `search(query, max_results)`：默认调用 [app/multiagent/research.py:20](app/multiagent/research.py:20) 的 `tavily_search()`。
2. `TavilySearch(max_results=5).ainvoke({"query": query})`。
3. Tavily 返回 dict，取 `results` 列表。
4. 把 `title`、`url`、`content` 格式化。
5. 用非流式模型生成带 URL 的答案。
6. 收集 URL 列表，返回 `SubagentResult`。

关键点：

- **联网信息可能冲突**：Tavily 结果可能来自不同时间，Research Agent 要明确说明不确定性。
- **不编造**：prompt 要求只根据搜索结果回答。
- **来源是 URL**：Supervisor 汇总时可以保留网页链接。
- **失败降级**：搜索失败时返回错误文本，不中断整个 MultiAgent。

## 7. Supervisor

### 7.1 构建

[app/multiagent/supervisor.py:93](app/multiagent/supervisor.py:93)：

```python
agent = create_agent(
    model or get_model(),
    tools=[ask_knowledge_agent, ask_research_agent],
    system_prompt=prompt,
    checkpointer=checkpointer,
    store=store,
)
```

两个工具：

- [app/multiagent/supervisor.py:116](app/multiagent/supervisor.py:116)：`ask_knowledge_agent`
- [app/multiagent/supervisor.py:128](app/multiagent/supervisor.py:128)：`ask_research_agent`

工具 docstring 会变成模型看到的工具描述：

```text
ask_knowledge_agent：向本地知识库 Agent 委派子问题……
ask_research_agent：向联网 Research Agent 委派子问题……
```

### 7.2 任务预算

[app/multiagent/supervisor.py:58](app/multiagent/supervisor.py:58)：

```python
@dataclass
class TaskBudget:
    max_tasks: int = 3
    max_per_agent: int = 2
```

`acquire()` 在锁内检查：

- 总调用数是否达到 3；
- 单个 Agent 是否达到 2；
- 通过后计数 +1。

这样即使 Supervisor 在同一个模型回合里并行调用两个工具，也不会超过预算。

### 7.3 Supervisor 提示词

[app/multiagent/supervisor.py:25](app/multiagent/supervisor.py:25) 附近的规则：

1. 判断需要本地知识、联网信息，还是两者都需要。
2. 如果两者都需要，尽量在同一轮并行调用两个工具。
3. 最多两轮任务分配；总共最多 3 次子任务；每个 Agent 最多 2 次。
4. 子 Agent 只拿到子问题，不共享完整对话。
5. 最终答案末尾必须有“来源”列表：至少本地文件路径和网页 URL。
6. 不编造；信息不足时明确说明。
7. Research Agent 对英文查询效果更好，版本/官方信息可保留英文关键词。

### 7.4 每次提问重建 Supervisor

[app/multiagent/supervisor.py:226](app/multiagent/supervisor.py:226) 的 `run_multiagent_cli()`：

```python
def make_agent():
    return build_multiagent(
        checkpointer=checkpointer,
        store=store,
        user_id=user_id,
    )
```

**为什么每次提问都重建 Agent？**

`TaskBudget` 是绑定在 Agent 实例上的。如果整个会话复用同一个 Agent，预算会跨问题累积，第二个问题可能直接被拒绝。

每次提问重建 Agent：

- 新问题有新预算；
- 长期记忆通过 Store 保留；
- 会话历史通过 checkpointer 保留。

## 8. 执行时序

用户问题：

> 结合本地知识库中 Agentic RAG 的重试策略，以及 2026 年 LangGraph 的最新版本信息，给我一个简短总结。

执行：

1. Supervisor 模型收到问题，看到两个工具 schema。
2. 模型生成两个 tool calls：
   - `ask_knowledge_agent`：本地知识库子问题；
   - `ask_research_agent`：联网子问题。
3. 两个工具分别 `acquire` 预算。
4. Knowledge Agent：
   - MCP `search_documents` → Chroma；
   - 非流式 LLM 总结；
   - 返回 `data/docs/...` 来源。
5. Research Agent：
   - Tavily 搜索；
   - 非流式 LLM 总结；
   - 返回网页 URL。
6. 两个 `ToolMessage` 回到 Supervisor。
7. Supervisor 生成最终答案，列出本地来源和网页 URL。
8. CLI 打印 `[Supervisor] 委派 ...` 和 `[Supervisor] ... 返回 ...`。

## 9. 实验结果

### 9.1 离线测试

```bash
uv run pytest -q
```

结果：

```text
24 passed, 6 skipped in 3.80s
```

Day 5 新增 3 个离线测试：

```text
tests/test_multiagent.py ...
3 passed
```

覆盖：

- `TaskBudget` 总数和单 Agent 限制；
- Knowledge Agent 返回答案和本地来源；
- Research Agent 返回答案和 URL。

### 9.2 真实 MultiAgent 测试

```bash
RUN_LIVE_TESTS=1 uv run pytest tests/test_multiagent_live.py -q
```

结果：

```text
1 passed in 15.95s
```

测试断言：

- Supervisor 的消息中存在 `ask_knowledge_agent` 调用；
- 存在 `ask_research_agent` 调用；
- 最终答案包含 `Agentic RAG`；
- 最终答案包含网页 URL（`http`）。

### 9.3 实际 CLI 输出

```bash
uv run python -m app.cli --mode multi \
  --question "结合本地知识库中 Agentic RAG 的重试策略，以及 2026 年 LangGraph 的最新版本信息，给我一个简短总结。" \
  --no-stream
```

关键轨迹：

```text
[Supervisor] 委派 ask_knowledge_agent {'question': '本地知识库中关于 Agentic RAG 的重试策略...'}
[Supervisor] 委派 ask_research_agent {'question': '2026 年 LangGraph 的最新版本是什么？...'}
[Supervisor] ask_knowledge_agent 返回: [knowledge] ...
[Supervisor] ask_research_agent 返回: [research] ...
```

最终答案同时包含：

- 本地来源：`data/docs/03-agentic-rag.md`
- 联网来源：`https://...`

### 9.4 单个子 Agent 验证

Knowledge Agent：

```text
Agentic RAG 在检索不足时最多重试 1 次改写、2 次检索。
来源：data/docs/03-agentic-rag.md
```

Research Agent：

```text
LangGraph 最新版本信息存在不一致，无法确定唯一版本。
来源：https://github.com/langchain-ai/langgraph/releases
```

这证明子 Agent 可以独立运行，也证明 Research Agent 不会在信息冲突时硬编造。

### 9.5 验收清单

| 要求 | 结果 |
| --- | --- |
| Supervisor 判断任务类型 | 通过 |
| 调用 Knowledge Agent | 通过（MCP） |
| 调用 Research Agent | 通过（Tavily） |
| Context Isolation | 通过（子 Agent 只拿子问题） |
| Result Aggregation | 通过（Supervisor 汇总） |
| 最多 3 次子任务 | 通过（TaskBudget） |
| 单个 Agent 最多 2 次 | 通过（TaskBudget） |
| 本地来源 + 网页 URL | 通过（最终答案包含） |

## 10. 遇到的问题

1. **Open Deep Research 源码无法直接拉取**  
   当前环境访问 GitHub raw 返回 `HTTP 000`，没有逐行阅读归档的 `multi_agent.py`。按计划描述的 Supervisor/Handoff/Context Isolation/Aggregation 模式实现最小版本，并在文档中说明。

2. **Supervisor 汇总时漏掉网页 URL**  
   第一次真实测试中，两个子 Agent 都被调用了，但 Supervisor 最终答案没有保留 Research Agent 的 URL，导致验收断言失败。修复：在 Supervisor 提示词中明确要求“最终答案末尾必须有来源列表，至少包含本地路径和网页 URL”。

3. **Tavily 结果可能互相冲突**  
   Research Agent 搜索 LangGraph 版本时得到不同来源的不同版本号。它明确说明“无法确认唯一版本”，而不是编造一个确定答案。这是正确行为，但也说明联网结果需要来源和时间校验。

4. **任务预算必须按“每次提问”重置**  
   如果整个会话复用一个 Supervisor，TaskBudget 会跨问题累积，后续问题可能被误拒绝。解决：每次提问重新构建 Supervisor，checkpointer 和 Store 保持会话状态。

5. **并行工具调用的预算竞争**  
   Supervisor 可能在一个模型回合里并行调用两个工具。`TaskBudget` 使用 `asyncio.Lock`，保证计数不会超卖。

6. **子 Agent 的上下文隔离**  
   子 Agent 只接收子问题，不接收 Supervisor 的完整历史。这降低了 prompt 长度和串扰，但也意味着 Supervisor 必须把子问题写清楚。

7. **异步 checkpointer**  
   MultiAgent 使用 `ainvoke`/`astream`，因此和 MCP 模式一样使用 `AsyncSqliteSaver`，不能使用同步 `SqliteSaver`。

8. **MCP 子进程启动开销**  
   Knowledge Agent 每次检索通过 MCP Client 启动 Server 子进程，单次调用有启动成本。后续可以复用持久 ClientSession。

## 11. 尚未理解/未实现

1. **精确的“最多两轮任务分配”**  
   当前用“最多 3 次子任务、单 Agent 最多 2 次”近似限制；没有在 Supervisor 图层面精确统计“轮次”。

2. **真正的并行调度与结果顺序**  
   Supervisor 可能并行调用两个工具，但结果到达顺序不固定；如何稳定合并、去重、冲突处理还没有深入研究。

3. **复杂 Supervisor 图**  
   当前是 `create_agent` + 工具 handoff。还没有实现多层 Supervisor、动态任务规划、Agent 之间的直接 handoff。

4. **引用一致性**  
   Supervisor 汇总时可能合并不同来源的冲突信息。还没有做引用对齐、事实一致性检查或来源可信度排序。

5. **成本与延迟控制**  
   一次 MultiAgent 请求可能包含：Supervisor 模型 + 2 个子 Agent 模型 + MCP 检索 + Tavily 搜索。还没有做并发限制、超时和成本预算。

6. **失败重试**  
   Tavily/MCP 失败时当前返回错误文本；还没有实现子 Agent 级别的重试、降级和熔断。

7. **MultiAgent 评估体系**  
   没有构建“需要本地 + 联网”的评估集，也没有量化任务拆分质量、子 Agent 命中率和最终答案完整性。

8. **记忆共享**  
   子 Agent 当前不读长期记忆；只有 Supervisor 的 system prompt 注入用户偏好。子 Agent 是否需要独立记忆，还没有设计。

## 今日交付

- 代码：
  - `app/multiagent/models.py`
  - `app/multiagent/knowledge.py`
  - `app/multiagent/research.py`
  - `app/multiagent/supervisor.py`
  - `app/cli.py`：新增 `--mode multi`
  - `app/multiagent/supervisor.py`：`AsyncSqliteSaver` 集成
- 测试：
  - `tests/test_multiagent.py`（离线 3 个）
  - `tests/test_multiagent_live.py`（真实 1 个）
- 验收：离线 24 passed / 6 skipped；真实 MultiAgent 1 passed；Supervisor 同时调用两个子 Agent 并汇总本地来源与网页 URL
- 交付标识：`v0.5-multiagent`（未创建 git tag；按约定不主动提交）
