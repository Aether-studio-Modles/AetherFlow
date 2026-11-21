"""推理服务层：把 :class:`~aetherflow.models.ServingEngine` 暴露成 HTTP。

这一层只做**协议与并发**，不碰模型：token 化交给 ``ByteTokenizer``，对话
模板交给 ``ChatTemplate``，调度交给 ``ServingEngine``。换任何一种都是替换
一个对象，不需要改服务代码。
"""

from .openai import (
    OpenAIService,
    RequestTimeout,
    create_server,
    serve,
)

__all__ = [
    "OpenAIService",
    "RequestTimeout",
    "create_server",
    "serve",
]
