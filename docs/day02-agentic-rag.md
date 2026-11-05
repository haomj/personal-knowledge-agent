# Day 2：Agentic RAG

> 交付版本：`v0.2-agentic-rag`  
> 运行环境：Python 3.12.15 / langgraph 1.2.15 / langchain 1.4.5 / langchain-chroma 1.1.0 / chromadb 1.5.9  
> 模型：DeepSeek `deepseek-flash`（生成、评分、改写）  
> Embedding：DashScope `text-embedding-v4`（1024 维）

## 1. 今日学习目标

今天要完成一个能检索本地知识库、判断检索质量、在必要时改写查询的 Agentic RAG：

1. 理解传统 RAG 与 Agentic RAG 的区别。
2. 实现 Markdown/TXT 文档加载、递归切分、Embedding 和 Chroma 向量存储。
3. 实现检索结果相关性评分（Retrieval Grading）。
4. 检索不足时改写查询并再次检索，最多重试一次。
5. 生成带来源引用的回答，并在 CLI 中单独展示来源。
6. 导入至少 10 篇技术文档，完成文档问答验收。

验收标准：Agent 能回答知识库中的问题，并显示来源；模糊或知识库没有的问题不会被编造。

## 2. Agentic RAG 核心概念

### 2.1 传统 RAG 与 Agentic RAG

传统 RAG 通常是线性流程：

```text
问题 -> 检索一次 -> 把结果交给模型 -> 回答
```

它的问题是一次检索失败后没有补救机会。Agentic RAG 在检索和生成之间加入“判断”和“重试”：

```text
用户问题
  -> 检索知识库
  -> 判断相关性
     -> 足够：生成带来源的回答
     -> 不足：改写查询
              -> 再次检索（最多一次）
              -> 生成回答
```

### 2.2 本项目的 State

Day 1 的 `AgentState` 只有 `messages`。Day 2 增加：

| 字段 | 作用 |
| --- | --- |
| `query` | 当前检索查询，可能被改写 |
| `documents` | 本次检索到的 `Document` 列表 |
| `sources` | 去重后的来源信息（路径、标题、分数） |
| `retrieval_attempts` | 已检索次数 |
| `retrieval_sufficient` | 最近一次评分是否足够 |
| `grade_reason` | 评分理由，便于调试 |

`messages` 仍然使用 `add_messages` reducer，保存完整对话历史；检索字段每次调用重新初始化，避免上一轮的资料串到下一轮。

### 2.3 为什么要评分

只看相似度分数不够。实测中，问“量子计算对 RAG 有什么影响”时，Chroma 仍然返回了 4 条文档，最高分 0.644，但内容完全不涉及量子计算。如果直接把结果交给生成模型，很容易产生“看起来有引用、其实答非所问”的回答。

评分节点让模型判断“资料是否足以回答”，不足时进入改写分支，而不是硬答。

## 3. 参考项目与源码位置

源码路径均相对于 `.venv/lib/python3.12/site-packages/`。

| 来源 | 位置 | 作用 |
| --- | --- | --- |
| 10 天计划 Day 2 | `10天AI-Agent源码学习与实战部署计划.pdf` | Agentic RAG 的流程图与验收标准 |
| LangGraph 条件边 | `langgraph/graph/state.py:982` | `add_conditional_edges`，RAG 路由的基础 |
| LangGraph 状态更新 | `langgraph/pregel/_algo.py:233` | `apply_writes` 把节点增量合并进 channel |
| 递归切分 | `langchain_text_splitters/character.py:91` | `RecursiveCharacterTextSplitter` |
| Chroma 封装 | `langchain_chroma/vectorstores.py:155` | `Chroma` 向量库 |
| Chroma 写入 | `langchain_chroma/vectorstores.py:597` | `add_texts` |
| Chroma 批量构建 | `langchain_chroma/vectorstores.py:1375` | `from_documents` |
| 相关性分数转换 | `langchain_chroma/vectorstores.py:904` | `_select_relevance_score_fn` |
| OpenAI 兼容 Embedding | `langchain_openai/embeddings/base.py:725` | `embed_documents` 批量调用 |
| Embedding 默认 batch | `langchain_openai/embeddings/base.py:260` | 默认 `chunk_size=1000`，DashScope 需要改成 10 |

计划里提到的 Agentic RAG LangGraph 参考项目，本次没有整体移植。我们按其核心设计复现了“检索 → 评分 → 改写 → 再检索 → 生成”的条件路由，保留了最小的可运行实现。

## 4. RAG Graph 构建流程

代码在 `app/rag/graph.py`。

```text
START
  -> retrieve
  -> grade
       ├─ sufficient=true  -> generate -> END
       └─ sufficient=false -> rewrite -> retrieve
                                  （最多 max_retries 次）
```

### 4.1 条件路由

```python
def route_after_grade(state: AgentState) -> str:
    if state.get("retrieval_sufficient"):
        return "generate"
    if int(state.get("retrieval_attempts", 0)) <= retry_limit:
        return "rewrite"
    return "generate"
```

当 `retry_limit=1` 时：

- 第一次检索后不足：`attempts=1 <= 1`，进入 `rewrite`；
- 第二次检索后仍不足：`attempts=2 <= 1` 为假，直接进入 `generate`，避免无限循环。

### 4.2 检索节点

```python
def retrieve_node(state):
    question = _last_human_text(state)
    query = clean_text(state.get("query") or question)
    attempts = int(state.get("retrieval_attempts", 0)) + 1
    documents = knowledge.retrieve(query)
    sources = sources_from_documents(documents)
    return {
        "query": query,
        "documents": documents,
        "sources": sources,
        "retrieval_attempts": attempts,
    }
```

每次检索都会覆盖 `documents` 和 `sources`，而不是追加。这样评分节点只看当前查询的结果。

### 4.3 生成节点

生成节点的系统提示词包含三部分：

1. 基础人设（来自 `app/config.py` 的 `system_prompt`）；
2. 资料是否足够的指令；
3. 检索到的资料和来源路径。

资料足够时：

```text
请只根据下面的资料回答，并在关键结论后用 [来源: 文件路径] 标注出处。
引用格式必须写成 [来源: data/docs/xxx.md]，不要只写 [1] 这样的编号。
```

资料不足时：

```text
请明确说明知识库中没有足够信息来确认答案，不要编造事实或来源；
如果资料中有一部分相关内容，可以说明它覆盖了什么、缺少什么。
```

### 4.4 可测试性

`build_rag_graph` 允许注入：

- `model`：生成节点使用的模型；
- `retriever`：检索器；
- `grader`：评分器；
- `rewriter`：改写器；
- `checkpointer`、`top_k`、`max_retries`。

单元测试因此不需要访问真实 API，也不需要 Chroma。

## 5. 文档加载与切分

### 5.1 加载

`app/rag/loader.py` 递归扫描 `data/docs/`，只支持：

- `.md`
- `.markdown`
- `.txt`

每个 `Document` 保留：

| metadata | 说明 |
| --- | --- |
| `source` | 相对项目根目录的路径，用于引用 |
| `relative_source` | 相对文档目录的路径 |
| `title` | 第一个 `# ` 标题 |
| `suffix` | 文件扩展名 |

加载时用 `clean_text()` 清洗非法 Unicode 代理项，避免 Day 1 遇到过的半截 UTF-8 问题。

### 5.2 切分

`app/rag/splitter.py` 使用 `RecursiveCharacterTextSplitter`：

```python
RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", " ", ""],
    add_start_index=True,
)
```

优先按 Markdown 二级/三级标题和段落切分，再退到中文句号、分号和字符。

每个 chunk 增加：

- `chunk_index`：同一文档内的序号；
- `chunk_id`：`source#index`，便于调试和去重。

实测 10 篇文档被切成 **22 个 chunk**。正文比较短，所以大多数文档 1～3 个 chunk。

## 6. Embedding 与 Chroma

### 6.1 Embedding 配置

DeepSeek 没有 Embedding API，因此按计划使用 DashScope 的 OpenAI 兼容接口：

```dotenv
DASHSCOPE_API_KEY=***
DASHSCOPE_BASE_URL=https://.../compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v4
```

实测 `text-embedding-v4` 和 `text-embedding-v3` 都可用，返回 **1024 维**向量。

### 6.2 DashScope 的 batch 限制

`OpenAIEmbeddings` 默认：

```python
chunk_size = 1000
```

第一次构建索引时报错：

```text
InvalidParameter: batch size is invalid, it should not be larger than 10
```

修复：

```python
OpenAIEmbeddings(
    ...,
    chunk_size=10,
)
```

这是供应商兼容层的重要差异，不能假设 OpenAI 的默认参数在所有兼容接口上都成立。

### 6.3 Chroma 配置

`app/rag/vector_store.py`：

```python
Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="data/chroma",
    collection_name="knowledge_base",
    collection_metadata={"hnsw:space": "cosine"},
)
```

- 文本检索使用 **cosine** 余弦距离。
- 索引持久化到 `data/chroma/`，该目录已加入 `.gitignore`，不提交。
- `reset=True` 会先删除旧目录，避免重复 chunk。
- 换 Embedding 模型时，向量维度可能变化，需要重建集合。

### 6.4 检索

`app/rag/retriever.py` 使用：

```python
store.similarity_search_with_relevance_scores(query, k=4)
```

把每条结果的 `score` 写回 metadata，并按来源去重：

- 同一个文件的多个 chunk 只保留最高分；
- 来源按分数从高到低排序；
- CLI 展示 `source` 和 `score`。

## 7. Retrieval Grading 与 Query Rewriting

### 7.1 评分模型

`app/rag/evaluator.py` 使用 Pydantic 模型：

```python
class RelevanceGrade(BaseModel):
    sufficient: bool
    reason: str
```

DeepSeek 的默认结构化输出不可用：

```text
This response_format type is unavailable now
```

改用 function calling：

```python
model.with_structured_output(
    RelevanceGrade,
    method="function_calling",
)
```

实测可以稳定返回 `sufficient` 和 `reason`。

### 7.2 评分策略

- 没有检索到任何文档：直接返回 `sufficient=False`，不调用模型。
- 有文档：把问题 + 前 4 条资料交给评分模型。
- 评分模型异常：降级为“最高分 >= 0.2 则通过”，并记录 warning。

评分不是单纯看分数。实测“量子计算”问题最高分 0.644，但语义完全不相关；LLM 评分正确判定为不足。

### 7.3 查询改写

`app/rag/rewriter.py` 的提示词要求：

- 保留专有名词、API 名称和数字；
- 补全代词指代；
- 不要回答问题；
- 只输出改写后的查询。

实测：

```text
from='量子计算对 RAG 有什么影响？'
to='量子计算对RAG的影响'
```

改写不一定能解决“知识库根本没有相关内容”的问题，但它能改善措辞、补全上下文，并在最多一次重试后停止。

### 7.4 路由结果

| 场景 | 检索次数 | 改写次数 | 生成 |
| --- | --- | --- | --- |
| 第一次评分足够 | 1 | 0 | 正常回答 |
| 第一次不足、第二次足够 | 2 | 1 | 正常回答 |
| 两次都不足 | 2 | 1 | 明确说明资料不足 |
| 没有文档 | 2 | 1 | 明确说明资料不足 |

## 8. 实验结果

### 8.1 索引构建

```bash
uv run python -m app.rag.ingest
```

结果：

```text
ingest.loaded documents=10 chunks=22
chroma.indexed chunks=22 collection=knowledge_base dir=data/chroma
索引完成：documents=10 chunks=22
```

### 8.2 离线测试

```bash
uv run pytest -q
```

结果：

```text
........s...ss...
14 passed, 3 skipped in 1.07s
```

覆盖：

- Day 1 基础图、多轮、流式、输入清洗；
- 文档加载与切分；
- 用假 Embedding 验证 Chroma 写入/检索；
- RAG 路由：一次足够、不足后改写、两次不足停止；
- 来源去重。

3 个 skipped 是真实 API 测试。

### 8.3 真实 RAG 端到端测试

```bash
RUN_LIVE_TESTS=1 uv run pytest tests/test_rag_live.py -q
```

结果：

```text
1 passed in 5.90s
```

测试会：

1. 用真实 DashScope Embedding 构建临时 Chroma；
2. 用 DeepSeek 做评分、生成；
3. 断言检索来源包含 `data/docs/03-agentic-rag.md`；
4. 断言最终回答非空。

### 8.4 验收问题一：Agentic RAG 重试次数

```bash
uv run python -m app.cli \
  --question "Agentic RAG 在检索结果不足时最多重试几次？"
```

关键日志：

```text
rag.retrieve query='Agentic RAG 在检索结果不足时最多重试几次？' docs=4 top_score=0.759
rag.node.grade sufficient=True reason=资料[1]的流程图明确写"改写查询 -> 再次检索（最多一次）"...
rag.node.generate sufficient=True docs=4
```

回答：

```text
Agentic RAG 在检索结果不足时，常见的重试上限是 1 次改写、2 次检索
（即最多改写查询一次并再次检索）[来源: data/docs/03-agentic-rag.md]。
```

来源：

```text
- data/docs/03-agentic-rag.md (score=0.759)
- data/docs/08-deepseek-api.md (score=0.564)
- data/docs/04-mcp.md (score=0.552)
```

### 8.5 验收问题二：DeepSeek 结构化输出

```bash
uv run python -m app.cli \
  --question "DeepSeek 的结构化输出有什么兼容性问题？"
```

回答要点：

- 默认 JSON schema `response_format` 不可用；
- 需要 `method="function_calling"`；
- 来源 `data/docs/08-deepseek-api.md`。

### 8.6 验收问题三：MCP 传输方式

```bash
uv run python -m app.cli \
  --question "MCP 的 stdio 和 Streamable HTTP 有什么区别？"
```

回答正确区分：

- stdio：Client 启动 Server 子进程，适合本地工具；
- Streamable HTTP：独立 HTTP 服务，适合远程/多客户端。

来源包括 `data/docs/04-mcp.md` 和 `data/docs/10-docker-deployment.md`。

### 8.7 不足与改写路径

```bash
uv run python -m app.cli \
  --question "量子计算对 RAG 有什么影响？"
```

实际执行：

```text
第一次检索：docs=4 top_score=0.644
评分：sufficient=False（资料完全没有提到量子计算）
改写：'量子计算对 RAG 有什么影响？' -> '量子计算对RAG的影响'
第二次检索：docs=4 top_score=0.611
评分：sufficient=False
生成：明确说明知识库没有足够信息
```

最终回答没有编造量子计算与 RAG 的关系，并说明现有资料只覆盖 Agentic RAG 流程、Embedding、Chroma 等内容。这是 Agentic RAG 评分节点的价值。

### 8.8 多轮 RAG

```bash
printf 'MCP 有哪些传输方式？\n其中 stdio 适合什么场景？\n/exit\n' | \
  uv run python -m app.cli --thread day2-multi
```

第二轮能结合第一轮上下文回答 stdio 的适用场景，并继续给出 `data/docs/04-mcp.md` 来源。

### 8.9 验收清单

| 要求 | 结果 |
| --- | --- |
| 导入 Markdown/TXT 文档 | 通过（10 篇） |
| 文档切分 | 通过（22 chunks） |
| Embedding | 通过（DashScope text-embedding-v4，1024 维） |
| 向量存储 | 通过（Chroma，cosine） |
| 检索相关文档 | 通过 |
| 生成带来源的回答 | 通过（`[来源: ...]` + 来源列表） |
| 检索不足最多重试一次 | 通过（单元测试 + 真实“量子计算”问题） |
| 多轮对话 | 通过 |

## 9. 遇到的问题

1. **DashScope Embedding batch 上限为 10**  
   `OpenAIEmbeddings` 默认 `chunk_size=1000`，第一次构建索引报 `batch size is invalid`。改成 `chunk_size=10` 后正常。

2. **DeepSeek 默认结构化输出不可用**  
   `with_structured_output(RelevanceGrade)` 使用 JSON schema `response_format`，供应商返回 `This response_format type is unavailable now`。显式改为 `method="function_calling"` 后可用。

3. **假 Embedding 的测试检索到了错误文档**  
   最初用字符哈希生成 8 维假向量，Chroma 把查询匹配到了 `a.md`。这不是 Chroma 的问题，而是假向量没有语义。测试改为二维关键词向量，只验证 Chroma 的写入/检索链路。

4. **相似度分数高不代表内容相关**  
   “量子计算”问题最高分 0.644，但仍然完全无关。因此必须有 Retrieval Grading，不能只靠分数阈值。

5. **模型引用格式不稳定**  
   有一次生成用了 `[1][2]` 而不是 `[来源: 文件路径]`。在生成提示词中明确“不要只写编号”后，引用格式更稳定；CLI 另外展示检索来源列表，保证来源可见。

6. **Chroma 索引是生成物**  
   `data/chroma/` 已加入 `.gitignore`，不提交；`data/docs/` 是知识库源文件，会提交。

7. **换 Embedding 模型必须重建集合**  
   向量维度由第一次写入决定。`text-embedding-v4` 是 1024 维，如果换成 2048 维模型，旧集合不能继续使用。

8. **IDE 类型告警：`api_key` 期望 `SecretStr`**  
   `ChatOpenAI` / `OpenAIEmbeddings` 的 `api_key` 参数类型是 `SecretStr | None | Callable[...]`，直接传 `str` 会被 PyCharm 标黄。改用 `SecretStr(settings.api_key)` 后告警消失；运行时行为不变，并且密钥不会出现在对象的 `repr` 中。验证方式：`uv run pytest -q` 仍为 14 passed / 3 skipped，真实 Chat 和 Embedding 调用都正常。

## 10. 尚未理解的知识点

1. **Chunking 质量评估**  
   目前固定 `chunk_size=600`、`overlap=100`。没有评估不同切分参数对召回率和答案质量的影响，也没有实现按标题层级的语义切分。

2. **Retrieval Grading 的准确率**  
   评分模型可能把“沾边”判为足够，也可能把“需要推理”判为不足。没有标注集，无法量化 precision/recall。

3. **Query Rewriting 与多轮上下文**  
   当前改写器只看到当前问题和当前查询，没有把最近几轮对话传进去。指代消解在复杂多轮场景下可能不够。

4. **来源过滤**  
   CLI 展示的是本次检索到的全部来源，而不是回答中实际引用的来源。后续可以解析回答里的 `[来源: ...]`，只展示被引用的文档。

5. **混合检索与重排序**  
   目前只有向量检索。关键词检索（BM25）、混合检索、cross-encoder rerank 都没有实现。

6. **Chroma 的持久化与并发**  
   单实例本地持久化已经工作，但多实例、并发写入、索引版本管理还没有研究。

7. **RAG 评估体系**  
   没有构建“问题 → 标准答案 → 应命中文档”的评估集，也没有计算 Recall@K、MRR、answer faithfulness。

8. **Agentic RAG LangGraph 参考项目的完整源码**  
   本次按计划复现了核心流程，但没有逐行阅读参考项目的实现，后续可以补读其查询改写、评分和路由细节。

## 今日交付

- 代码：
  - `app/rag/loader.py`、`splitter.py`、`embeddings.py`、`vector_store.py`
  - `app/rag/retriever.py`、`evaluator.py`、`rewriter.py`
  - `app/rag/graph.py`、`app/rag/ingest.py`
  - `app/cli.py`（RAG / basic 两种模式）
  - `app/config.py`（Embedding 与 RAG 配置）
  - `app/state.py`（RAG State 字段）
- 知识库：`data/docs/` 10 篇 Markdown
- 测试：`tests/test_rag_pipeline.py`、`test_rag_graph.py`、`test_rag_live.py`
- 验收：离线 14 passed / 3 skipped；真实 RAG 1 passed；10 篇文档、22 chunks；带来源回答与不足重试均通过
- 交付标识：`v0.2-agentic-rag`（未创建 git tag；按约定不主动提交）
