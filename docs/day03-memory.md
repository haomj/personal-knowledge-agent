# Day 3：Memory 与持久化

> 交付版本：`v0.3-memory`  
> 运行环境：Python 3.12.15 / langgraph 1.2.15 / langgraph-checkpoint-sqlite 3.1.3  
> 短期记忆：`SqliteSaver`（`data/checkpoints.sqlite`）  
> 长期记忆：`SqliteStore`（`data/memory.sqlite`）

## 1. 今日学习目标

今天要完成两个层次的记忆：

1. **短期记忆**：用 SQLite checkpointer 按 `thread_id` 保存会话，进程重启后恢复。
2. **长期记忆**：用 LangGraph Store 按 `user_id` 保存用户偏好，不同 thread 共享，不同用户隔离。

验收标准：

- 同一个 `thread_id` 在重启后能恢复历史；
- 不同 thread 的同一用户能读到相同偏好；
- 不同用户之间记忆隔离；
- RAG 模式也能使用这两层记忆。

## 2. Memory 核心概念

| 概念 | 含义 |
| --- | --- |
| Thread | 一次会话的标识，`thread_id` 是 checkpoint 的主键 |
| Checkpoint | 某个 superstep 后的完整状态快照 |
| State History | 一个 thread 下按时间排列的多个 checkpoint |
| Store | 跨 thread、跨会话的长期数据存储 |
| Namespace | Store 中的层级路径，例如 `("users", "adrian", "preferences")` |
| Short-term Memory | 当前会话的上下文，主要靠 checkpointer |
| Long-term Memory | 用户偏好、项目背景等跨会话事实，主要靠 Store |

### 2.1 短期记忆

```text
Thread A -> Conversation -> Checkpoint -> SQLite
```

同一个 `thread_id` 的每次调用都会读写同一份 checkpoint。不同 `thread_id` 之间互不影响。

### 2.2 长期记忆

```text
User -> /remember -> Store -> 下一轮生成时注入 system prompt
```

长期记忆不依赖某个 thread。同一个用户开新会话，仍然能读到之前保存的偏好。

### 2.3 为什么要分开

- 短期记忆回答“这次对话刚才说了什么”。
- 长期记忆回答“这个用户长期以来的偏好和背景是什么”。

把两者混在一个线程里，会导致跨会话信息丢失；把长期偏好塞进每个 checkpoint，又会重复存储、难以更新。

## 3. 阅读的源码

源码路径相对于 `.venv/lib/python3.12/site-packages/`。

| 文件 | 关键位置 | 作用 |
| --- | --- | --- |
| `langgraph/checkpoint/base/__init__.py` | `BaseCheckpointSaver` 199 行；`get_tuple` 262 行；`put` 301 行；`put_writes` 324 行；`CheckpointTuple` 140 行 | checkpointer 抽象接口 |
| `langgraph/checkpoint/sqlite/__init__.py` | `SqliteSaver` 49 行；`from_conn_string` 104 行；`get_tuple` 210 行；`list` 311 行；`put` 400 行；`put_writes` 458 行 | SQLite 实现 |
| `langgraph/store/base/__init__.py` | `Item` 51 行；`BaseStore` 708 行；`get` 756 行；`search` 784 行；`put` 866 行；`delete` 952 行 | Store 抽象接口 |
| `langgraph/store/sqlite/base.py` | `SqliteStore` 789 行；`from_conn_string` 1014 行；`setup` 1062 行 | SQLite Store 实现 |

### 3.1 BaseCheckpointSaver

`CheckpointTuple` 包含：

```python
class CheckpointTuple(NamedTuple):
    config: RunnableConfig
    checkpoint: Checkpoint
    metadata: CheckpointMetadata
    parent_config: RunnableConfig | None = None
    pending_writes: list[PendingWrite] | None = None
```

`get_tuple()` 按 `thread_id` + `checkpoint_ns` + 可选 `checkpoint_id` 读取 checkpoint；`put()` 保存状态快照；`put_writes()` 保存某个任务产生的中间写入。

### 3.2 SqliteSaver

`SqliteSaver.from_conn_string("data/checkpoints.sqlite")` 内部使用：

```python
sqlite3.connect(conn_string, check_same_thread=False)
```

`check_same_thread=False` 是因为 LangGraph 可能在线程池里执行节点；`SqliteSaver` 自己用锁保证写入安全。

文档也明确说明：`SqliteSaver` 适合轻量、同步、单实例场景，不适合多实例高并发。

### 3.3 BaseStore 与 SqliteStore

Store 的核心接口：

- `get(namespace, key)`：读取单个 Item。
- `put(namespace, key, value)`：写入一个 dict。
- `search(namespace_prefix, ...)`：按 namespace 前缀搜索。
- `delete(namespace, key)`：删除。

`SqliteStore.from_conn_string()` 使用 `isolation_level=None` 的自动提交连接；`setup()` 在第一次使用时创建表和迁移。

本项目只用了 `get` 和 `put`，没有使用 `search` 的向量检索能力。

## 4. 短期记忆流程

### 4.1 Thread -> Checkpoint

CLI 每次调用图时传入：

```python
config = {"configurable": {"thread_id": "day3-persist"}}
```

LangGraph 在 superstep 结束后调用 checkpointer：

1. `put()` 保存 checkpoint；
2. `put_writes()` 保存 pending writes；
3. 下一次同 thread 调用时，先用 `get_tuple()` 恢复 State。

### 4.2 重启恢复

实测两个独立进程：

第一个进程：

```text
thread_id=day3-persist
agent.start messages=1
用户：我叫 Adrian，请记住。
助手：好的，Adrian，我记住了。
```

第二个进程，使用同一个 `thread_id`：

```text
agent.start messages=3
用户：我叫什么名字？
助手：Adrian
```

`messages=3` 说明恢复了两条历史消息（Human + AI），加上本轮 Human。进程重启后历史没有丢。

### 4.3 为什么需要 thread_id

`thread_id` 是 checkpoint 的主键。没有它，checkpointer 无法知道该读哪份状态；换一个 `thread_id`，就是另一段独立会话。

### 4.4 SQLite 的边界

SQLite 适合：

- 本地学习；
- 单实例部署；
- 低并发。

不适合：

- 多实例共享状态；
- 高并发写入；
- 跨机器恢复。

多实例需要 Postgres 等共享存储。

## 5. 长期记忆流程

### 5.1 Namespace

本项目使用：

```python
("users", user_id, "preferences")
```

例如：

```python
("users", "adrian", "preferences")
("users", "bob", "preferences")
```

不同用户的 namespace 不同，所以互不可见。namespace label 不能包含 `.`，代码里会把 `.` 替换成 `_`。

### 5.2 显式写入

第一版不做自动记忆提取，只通过 CLI 命令写入：

```text
/remember project_code=ZK-2026
```

保存后的 Store 结构：

```python
namespace = ("users", "adrian", "preferences")
key = "preferences"
value = {
    "language": "中文",
    "project_code": "ZK-2026",
}
```

### 5.3 读取与注入

基础图和 RAG 图的生成节点都会调用：

```python
preference_prompt(store, user_id)
```

如果有偏好，就拼到 system prompt：

```text
用户长期偏好（来自 Store）：
- language: 中文
- project_code: ZK-2026
```

这样模型在生成时可以遵循用户偏好，而不需要把偏好写进每个 checkpoint。

### 5.4 跨 thread 与用户隔离

实测：

- adrian 在 thread `day3-code-a` 保存 `project_code=ZK-2026`；
- adrian 在另一个 thread `day3-code-b` 问“我的项目代号是什么”，回答 `ZK-2026`，日志 `memory=True`；
- bob 在 thread `day3-code-c` 问同样的问题，回答“我无法确定你的项目代号”，日志 `memory=False`。

这说明长期记忆按用户共享、按用户隔离，而不是按 thread 隔离。

## 6. 代码实现

### 6.1 短期记忆

`app/memory/checkpointer.py`：

```python
@contextmanager
def open_checkpointer(db_path):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as checkpointer:
        yield checkpointer
```

CLI 用 `with` 包住整个会话，保证数据库连接在对话期间一直打开。

### 6.2 长期记忆

`app/memory/store.py` 提供：

- `open_store(db_path)`：打开 `SqliteStore`；
- `user_namespace(user_id)`：生成用户 namespace；
- `load_preferences(store, user_id)`：读取偏好；
- `save_preference(store, user_id, key, value)`：保存偏好；
- `delete_preference(...)`：删除偏好；
- `format_preferences(preferences)`：生成 prompt 文本；
- `preference_prompt(store, user_id)`：节点调用入口。

### 6.3 State

`AgentState` 新增：

```python
user_id: str
```

每次调用图时由 CLI 传入当前用户。节点从 State 读取 `user_id`，再访问对应的 Store namespace。

### 6.4 图

- `build_graph(..., store=store)`：基础图在 `agent_node` 中注入偏好。
- `build_rag_graph(..., store=store)`：RAG 图在 `generate_node` 中注入偏好。
- `build_chat_graph(checkpointer=..., store=...)`：CLI 基础模式入口。

没有传 `store` 时，行为与 Day 1/2 完全一致；测试可以继续使用 `InMemorySaver`。

### 6.5 CLI

新增参数：

```text
--user adrian
--thread day3-chat
--checkpoint-db data/checkpoints.sqlite
--memory-db data/memory.sqlite
```

新增命令：

```text
/remember key=value
/preferences
/new
/exit
```

启动时打印当前 `user_id`、`thread_id`、数据库路径，方便确认状态。

## 7. 实验结果

### 7.1 离线测试

```bash
uv run pytest -q
```

结果：

```text
............s...ss...
18 passed, 3 skipped in 0.86s
```

Day 3 新增 4 个记忆测试：

```text
tests/test_memory.py ....
4 passed in 0.68s
```

覆盖：

1. SQLite checkpointer 重启后恢复同一 thread；
2. SQLite Store 重启后偏好仍在；
3. 同一用户跨 thread 共享偏好、不同用户隔离；
4. RAG 生成节点也能收到用户偏好。

### 7.2 短期记忆重启验证

第一个进程：

```bash
printf '我叫 Adrian，请记住。\n/exit\n' | \
  uv run python -m app.cli --mode basic --thread day3-persist --user adrian
```

第二个进程：

```bash
printf '我叫什么名字？只回答名字。\n/exit\n' | \
  uv run python -m app.cli --mode basic --thread day3-persist --user adrian
```

关键日志：

```text
checkpointer=SqliteSaver
agent.start messages=3
助手：Adrian
```

### 7.3 长期记忆验证

保存：

```text
/remember project_code=ZK-2026
已记住：project_code = ZK-2026
```

同一用户、不同 thread：

```text
agent.start messages=1 user_id=adrian memory=True
助手：ZK-2026
```

不同用户：

```text
agent.start messages=1 user_id=bob memory=False
助手：我无法确定你的项目代号，因为对话中没有提供相关信息。
```

> 第一次测试用 `language=中文` 时，bob 也回答了“中文”。这不是记忆泄漏，而是问题本身是中文，模型做了猜测。换成不可推断的 `project_code=ZK-2026` 后，隔离效果才被可靠验证。

### 7.4 RAG + SQLite

```bash
uv run python -m app.cli \
  --question "Agentic RAG 在检索结果不足时最多重试几次？" \
  --thread day3-rag-persist --user adrian
```

关键日志：

```text
checkpointer=SqliteSaver store=SqliteStore
rag.node.generate sufficient=True docs=4 user_id=adrian memory=True
```

第二个进程使用同一个 thread 追问“那它的查询改写是怎么做的？”，能恢复上一轮对话并继续 RAG 检索，说明 RAG State 里的 `Document`、`sources` 等字段也能被 SQLite 序列化和恢复。

### 7.5 验收清单

| 要求 | 结果 |
| --- | --- |
| 同一 thread 连续提问 | 通过 |
| 重启后恢复历史 | 通过（两个独立进程） |
| 不同 thread 共享同一用户记忆 | 通过 |
| 不同用户记忆隔离 | 通过 |
| RAG 模式使用记忆 | 通过 |
| 本地数据不进入 git | 通过（`data/*.sqlite*` 已忽略） |

## 8. 遇到的问题

1. **SQLite 连接的线程参数**  
   LangGraph 可能在线程池中执行节点，`sqlite3.connect` 需要 `check_same_thread=False`。`SqliteSaver.from_conn_string` 已经处理；自定义连接时要自己加。

2. **数据库连接的生命周期**  
   不能在每次提问后关闭 checkpointer，否则下一轮无法读取状态。CLI 用 `with open_store(...), open_checkpointer(...)` 包住整个对话循环。

3. **namespace label 不能含 `.`**  
   Store 会校验 namespace label。代码中把 `user_id` 里的 `.` 替换成 `_`，避免用户邮箱之类的 id 导致异常。

4. **Store 的 value 必须是 dict**  
   `store.put(namespace, key, value)` 的 value 是 JSON 可序列化的 mapping。不要把 `Document` 或自定义对象直接塞进 Store；本项目只存字符串偏好。

5. **用可推断的值测试隔离会误判**  
   `language=中文` 时，模型即使没有记忆也会猜“中文”。后来改用 `project_code=ZK-2026`，才真正验证了用户隔离。测试设计要避免模型能靠常识猜到的值。

6. **RAG State 的序列化**  
   RAG State 中包含 `Document` 对象，`SqliteSaver` 使用 `JsonPlusSerializer` 能正常序列化；但如果以后换成自定义对象，需要确认 serde 支持。

7. **本地数据库包含用户数据**  
   `data/checkpoints.sqlite`、`data/memory.sqlite` 已加入 `.gitignore`。生产环境还需要考虑加密、访问控制和备份。

8. **第一版没有自动记忆提取**  
   只支持 `/remember` 显式写入。自动从对话中提取偏好需要额外的抽取、去重、冲突处理和审核，留到后续。

## 9. 尚未理解的知识点

1. **State History 与时间旅行**  
   还没有实际使用 `graph.get_state_history(config)` 查看历史 checkpoint，也没有尝试从旧 checkpoint 分支恢复。

2. **Checkpoint 的内部版本机制**  
   `channel_versions`、`pending_writes`、`parent_config` 在恢复和重放中的具体作用，需要结合 Day 6 的 interrupt 再深入。

3. **Store 的 search 与向量索引**  
   目前只用了 `get`/`put`。`SqliteStore` 支持索引和 `search`，但还没有实验语义搜索、TTL、filter 等能力。

4. **LangMem 自动记忆提取**  
   计划里提到 LangMem；目前没有实现自动提取、记忆去重和冲突合并。

5. **多实例持久化**  
   SQLite 只适合单实例。多实例需要 Postgres checkpointer/store，还要处理并发、锁和迁移。

6. **记忆质量与隐私**  
   长期记忆应该保存什么、保存多久、如何让用户删除，都是产品级问题；当前只是最小实现。

7. **Thread 管理**  
   还没有实现“列出我的所有 thread”“重命名 thread”“删除 thread”等管理功能。

## 今日交付

- 代码：
  - `app/memory/checkpointer.py`
  - `app/memory/store.py`
  - `app/graph.py`（基础图偏好注入）
  - `app/rag/graph.py`（RAG 图偏好注入）
  - `app/cli.py`（SQLite、`--user`、`/remember`、`/preferences`）
  - `app/config.py`（checkpoint/memory 数据库配置）
  - `app/state.py`（`user_id`）
- 测试：`tests/test_memory.py`（4 个记忆测试）
- 数据：`data/checkpoints.sqlite`、`data/memory.sqlite`（本地生成，不提交）
- 验收：离线 18 passed / 3 skipped；重启恢复、跨 thread 共享、用户隔离、RAG + Memory 均通过
- 交付标识：`v0.3-memory`（未创建 git tag；按约定不主动提交）
