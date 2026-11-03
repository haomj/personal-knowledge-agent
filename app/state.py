"""Graph 的共享状态定义。"""

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Day 1 的 Agent 状态：只有一条消息历史。

    MessagesState 已经定义为：

        class MessagesState(TypedDict):
            messages: Annotated[list[AnyMessage], add_messages]

    add_messages 是 reducer：节点返回 {"messages": [新消息]} 时，
    LangGraph 会把它追加到已有列表，而不是覆盖整个列表。
    """
