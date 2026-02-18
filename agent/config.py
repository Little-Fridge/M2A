from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Type, TypeVar, Any

T = TypeVar("T")
TIME_FMT = r"%Y-%m-%d %H:%M"

def _from_dict(cls: Type[T], data: dict[str, Any]) -> T:
    known = {f.name: f for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, val in data.items():
        if key not in known:
            continue
        f = known[key]

        if isinstance(val, dict) and hasattr(f.type, "__dataclass_fields__"):
            val = _from_dict(f.type, val)
        kwargs[key] = val
    return cls(**kwargs)

def _override_from_env(obj: T, prefix: str) -> T:
    updates: dict[str, Any] = {}
    for f in fields(obj):
        env_key = f"{prefix}_{f.name.upper()}"
        raw = os.environ.get(env_key)
        if raw is None:
            continue

        origin_type = f.type

        if hasattr(origin_type, "__dataclass_fields__"):
            continue
        try:
            if origin_type is bool or origin_type == "bool":
                updates[f.name] = raw.lower() in ("1", "true", "yes")
            elif origin_type is int or origin_type == "int":
                updates[f.name] = int(raw)
            elif origin_type is float or origin_type == "float":
                updates[f.name] = float(raw)
            else:
                updates[f.name] = raw
        except (ValueError, TypeError):
            pass  
    if not updates:
        return obj
    return _from_dict(type(obj), {**asdict(obj), **updates}) 


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    api_key: str = "EMPTY"
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0
    max_tokens: int = 1200
    timeout: int = 20


@dataclass
class TextEmbeddingConfig:
    api_key: str = "EMPTY"
    api_base: str = "http://localhost:8100/v1"
    model: str = "text-embedding-3-small"


@dataclass
class MultimodalEmbeddingConfig:
    api_key: str = "EMPTY"
    api_base: str = "http://localhost:8050/v1"
    model: str = "siglip2-base-patch16-384"

@dataclass
class MemoryManagerConfig:
    context_window: int = 5
    max_iteration: int = 5


@dataclass
class MemoryConfig:
    raw_db_path: str = "raw_messages.db"
    semantic_db_path: str = "semantic_store.db"
    reuse_db: bool = False
    max_raw_messages_return: int = 20


@dataclass
class ChatAgentConfig:
    max_chat_context: int = 8000
    max_query_iteration: int = 5
    max_update_iteration: int = 5


@dataclass
class M2AConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    text_embedding: TextEmbeddingConfig = field(default_factory=TextEmbeddingConfig)
    multimodal_embedding: MultimodalEmbeddingConfig = field(default_factory=MultimodalEmbeddingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    chat_agent: ChatAgentConfig = field(default_factory=ChatAgentConfig)
    memory_manager: MemoryManagerConfig = field(default_factory=MemoryManagerConfig)

    @classmethod
    def from_env(cls) -> "M2AConfig":
        cfg = cls()
        cfg.llm = _override_from_env(cfg.llm, "M2A_LLM")
        cfg.text_embedding = _override_from_env(cfg.text_embedding, "M2A_TEXT_EMBEDDING")
        cfg.multimodal_embedding = _override_from_env(cfg.multimodal_embedding, "M2A_MULTIMODAL_EMBEDDING")
        cfg.memory = _override_from_env(cfg.memory, "M2A_MEMORY")
        cfg.evaluation = _override_from_env(cfg.evaluation, "M2A_EVALUATION")
        cfg.chat_agent = _override_from_env(cfg.chat_agent, "M2A_CHAT_AGENT")
        return cfg

    @classmethod
    def from_file(cls, config_path: str) -> "M2AConfig":
        import tomllib
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'rb') as f:
            data = tomllib.load(f)
        return cls._from_dict_nested(data)

    @classmethod
    def from_file_and_env(cls, config_path: str) -> "M2AConfig":
        cfg = cls.from_file(config_path)
        cfg.llm = _override_from_env(cfg.llm, "M2A_LLM")
        cfg.text_embedding = _override_from_env(cfg.text_embedding, "M2A_TEXT_EMBEDDING")
        cfg.multimodal_embedding = _override_from_env(cfg.multimodal_embedding, "M2A_MULTIMODAL_EMBEDDING")
        cfg.memory = _override_from_env(cfg.memory, "M2A_MEMORY")
        cfg.evaluation = _override_from_env(cfg.evaluation, "M2A_EVALUATION")
        cfg.chat_agent = _override_from_env(cfg.chat_agent, "M2A_CHAT_AGENT")
        return cfg

    @classmethod
    def _from_dict_nested(cls, data: dict) -> "M2AConfig":
        sub_classes = {
            "llm": LLMConfig,
            "text_embedding": TextEmbeddingConfig,
            "multimodal_embedding": MultimodalEmbeddingConfig,
            "memory": MemoryConfig,
            "chat_agent": ChatAgentConfig,
            "memory_manager": MemoryManagerConfig
        }
        cfg = cls()
        for key, sub_cls in sub_classes.items():
            if key in data:
                current = asdict(getattr(cfg, key))
                current.update(data[key]) 
                setattr(cfg, key, _from_dict(sub_cls, current))
        return cfg

    def save_to_file(self, config_path: str) -> None:
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def to_dict(self) -> dict:
        return asdict(self)

def get_default_config() -> M2AConfig:
    return M2AConfig()


def get_config_from_env() -> M2AConfig:
    return M2AConfig.from_env()
