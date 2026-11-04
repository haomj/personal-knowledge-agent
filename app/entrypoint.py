"""运行时入口：LangGraph Server / langgraph.json 从这里加载 graph。

Day 2 默认暴露 Agentic RAG 图；同时保留 Day 1 的基础图。
注意：这个模块在导入时就会创建模型和向量库客户端，因此必须能读到 .env。
CLI 和测试不要导入它，直接使用 build_rag_graph / build_graph。
"""

from app.graph import build_graph
from app.rag.graph import build_rag_graph

graph = build_rag_graph()
basic_graph = build_graph()
