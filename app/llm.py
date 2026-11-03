"""兼容旧入口：模型工厂已迁移到 app.config.get_model。

保留这个模块是为了不破坏早期示例中的导入习惯；
新代码请直接从 app.config 导入。
"""

from app.config import get_model

__all__ = ["get_model"]
