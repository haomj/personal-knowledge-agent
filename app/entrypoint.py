"""运行时入口：LangGraph Server / langgraph.json 从这里加载 graph。

注意：这个模块在导入时就会创建模型，因此必须能读到 .env。
CLI 和测试不要导入它，直接使用 app.graph.build_graph 即可。
"""

from app.graph import build_graph

graph = build_graph()
