"""应用配置与模型工厂。

所有环境变量都在这里读取，避免业务代码里散落 os.getenv 调用。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

# 项目根目录下的 .env 在导入配置时加载一次。
load_dotenv()

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL_NAME = "deepseek-flash"
DEFAULT_SYSTEM_PROMPT = (
    "你是一个个人知识研究助手。回答要准确、简洁；"
    "如果信息不足，请明确说明，不要编造事实或来源。"
)


@dataclass(frozen=True)
class Settings:
    """一次进程运行期间不会变化的配置。"""

    model_name: str
    base_url: str
    api_key: str
    temperature: float
    thinking: str
    system_prompt: str
    log_level: str
    embedding_model: str
    embedding_base_url: str
    embedding_api_key: str
    docs_dir: str
    chroma_dir: str
    chroma_collection: str
    rag_top_k: int
    rag_max_retries: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_name=os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME),
            base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            # DeepSeek 默认开启思考模式；Day 1 只做基础问答，关掉可以减少等待和费用。
            thinking=os.getenv("LLM_THINKING", "disabled").lower(),
            system_prompt=os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
            embedding_base_url=os.getenv(
                "EMBEDDING_BASE_URL",
                os.getenv(
                    "DASHSCOPE_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
            ),
            embedding_api_key=os.getenv(
                "EMBEDDING_API_KEY", os.getenv("DASHSCOPE_API_KEY", "")
            ),
            docs_dir=os.getenv("DOCS_DIR", "data/docs"),
            chroma_dir=os.getenv("CHROMA_DIR", "data/chroma"),
            chroma_collection=os.getenv("CHROMA_COLLECTION", "knowledge_base"),
            rag_top_k=int(os.getenv("RAG_TOP_K", "4")),
            rag_max_retries=int(os.getenv("RAG_MAX_RETRIES", "1")),
        )


@lru_cache
def get_settings() -> Settings:
    """缓存配置，保证同一进程内只解析一次环境变量。"""
    return Settings.from_env()


def get_model(
    settings: Settings | None = None, *, streaming: bool = True
) -> ChatOpenAI:
    """创建 DeepSeek 的 OpenAI 兼容模型客户端。

    DeepSeek 官方接口兼容 OpenAI Chat Completions：
    base_url=https://api.deepseek.com，模型名 deepseek-flash。

    streaming=True 时，即便调用 .invoke()，模型内部也会走流式 HTTP，
    这样 LangGraph 的 stream_mode="messages" 才能拿到逐 token 的回调。
    """
    settings = settings or get_settings()
    if not settings.api_key:
        raise RuntimeError(
            "缺少 DEEPSEEK_API_KEY。请在项目根目录的 .env 中配置后重试。"
        )

    extra_body = None
    if settings.thinking in {"enabled", "disabled"}:
        extra_body = {"thinking": {"type": settings.thinking}}

    return ChatOpenAI(
        model=settings.model_name,
        base_url=settings.base_url,
        # LangChain 的参数类型是 SecretStr；同时避免密钥出现在对象 repr 中。
        api_key=SecretStr(settings.api_key),
        temperature=settings.temperature,
        streaming=streaming,
        stream_usage=True,
        extra_body=extra_body,
    )


def get_embeddings(settings: Settings | None = None) -> OpenAIEmbeddings:
    """创建 Embedding 客户端。

    Day 2 使用 DashScope 的 OpenAI 兼容 Embedding 接口。
    DeepSeek 本身没有 Embedding API，所以向量化走 DashScope。
    """
    settings = settings or get_settings()
    if not settings.embedding_api_key:
        raise RuntimeError(
            "缺少 DASHSCOPE_API_KEY（或 EMBEDDING_API_KEY）。"
            "请在项目根目录的 .env 中配置后重试。"
        )
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        base_url=settings.embedding_base_url,
        # 与 ChatOpenAI 一样，类型检查期望 SecretStr。
        api_key=SecretStr(settings.embedding_api_key),
        check_embedding_ctx_length=False,
        # DashScope 兼容接口单次最多接受 10 条文本，超过会报 batch size invalid。
        chunk_size=10,
    )


def configure_logging(level: str | None = None) -> None:
    """配置根 logger；日志输出到 stderr，不干扰终端里的流式答案。"""
    settings = get_settings()
    resolved = (level or settings.log_level).upper()
    logging.basicConfig(
        level=getattr(logging, resolved, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    # 第三方库的 INFO 日志（代理探测、每次 HTTP 请求）对 Day 1 的观察没有帮助。
    for noisy in ("httpx", "httpx2", "httpcore", "langchain_openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
