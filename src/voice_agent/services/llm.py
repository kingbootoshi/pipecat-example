import os
from typing import Mapping, Any

from pipecat.services.openrouter.llm import OpenRouterLLMService


def make_llm(llm_config: Mapping[str, Any] | None = None) -> OpenRouterLLMService:
    cfg = dict(llm_config or {})
    return OpenRouterLLMService(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=cfg.get("model", "anthropic/claude-haiku-4.5"),
    )


__all__ = ["make_llm"]

