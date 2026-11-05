"""Embedding 工厂。

Day 2 使用 DashScope 的 OpenAI 兼容 Embedding 接口，
具体配置集中在 app/config.py，避免业务代码散落环境变量读取。
"""

from app.config import get_embeddings

__all__ = ["get_embeddings"]
