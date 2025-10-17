import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

# OpenPipe provides an OpenAI-compatible client
from openpipe import OpenAI  # type: ignore


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class AIClient:
    """
    Centralized AI client using OpenPipe SDK while routing calls to OpenRouter.

    - Expects OPENROUTER_API_KEY for auth to OpenRouter
    - Optionally uses OPENPIPE_API_KEY for OpenPipe telemetry/tracing
    - Routes to OpenRouter via base_url: https://openrouter.ai/api/v1
    - Supports OpenRouter attribution headers (HTTP-Referer, X-Title)

    This module is intentionally decoupled from the main voice agent pipeline.
    """

    def __init__(
        self,
        *,
        openrouter_api_key: str,
        openpipe_api_key: Optional[str] = None,
        openrouter_base_url: str = "https://openrouter.ai/api/v1",
        openpipe_base_url: Optional[str] = None,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
    ) -> None:
        openpipe_cfg: Dict[str, Any] = {}
        if openpipe_api_key:
            openpipe_cfg["api_key"] = openpipe_api_key
        if openpipe_base_url:
            openpipe_cfg["base_url"] = openpipe_base_url

        # The OpenPipe OpenAI client accepts the same kwargs as OpenAI
        # We set base_url to OpenRouter and pass OpenPipe telemetry config
        self._client = OpenAI(
            api_key=openrouter_api_key,
            base_url=openrouter_base_url,
            openpipe=openpipe_cfg or None,  # type: ignore[arg-type]
        )

        # Base attribution headers for OpenRouter leaderboards, optional
        extra_headers: Dict[str, str] = {}
        if site_url:
            extra_headers["HTTP-Referer"] = site_url
        if site_name:
            extra_headers["X-Title"] = site_name
        self._base_headers = extra_headers

    @classmethod
    def from_env(cls) -> "AIClient":
        return cls(
            openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
            openpipe_api_key=os.environ.get("OPENPIPE_API_KEY"),
            openrouter_base_url=os.environ.get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            openpipe_base_url=os.environ.get("OPENPIPE_BASE_URL"),
            site_url=os.environ.get("OPENROUTER_SITE_URL"),
            site_name=os.environ.get("OPENROUTER_SITE_NAME"),
        )

    def chat(
        self,
        *,
        messages: Iterable[ChatMessage] | Iterable[Mapping[str, str]],
        model: str = "openai/gpt-4o-mini",
        metadata: Optional[Mapping[str, Any]] = None,
        temperature: Optional[float] = None,
        response_format: Optional[Mapping[str, Any]] = None,
        extra_headers: Optional[MutableMapping[str, str]] = None,
        tools: Optional[list[Mapping[str, Any]]] = None,
        tool_choice: Optional[Mapping[str, Any] | str] = None,
        parallel_tool_calls: Optional[bool] = None,
        stream: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Perform a basic non-streaming chat completion.

        Returns the full OpenAI-compatible response dict.
        """
        msgs = [m if isinstance(m, dict) else {"role": m.role, "content": m.content} for m in messages]
        headers: Dict[str, str] = dict(self._base_headers)
        if extra_headers:
            headers.update(extra_headers)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": msgs,
        }
        if metadata is not None:
            payload["metadata"] = dict(metadata)
        if response_format is not None:
            payload["response_format"] = dict(response_format)
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if tools is not None:
            payload["tools"] = list(tools)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = bool(parallel_tool_calls)
        if stream is not None:
            payload["stream"] = bool(stream)

        return self._client.chat.completions.create(  # type: ignore[no-any-return]
            extra_headers=headers,
            **payload,
        )

    # NOTE: This codebase intentionally does NOT support JSON-schema-based
    # `response_format` structured outputs as a convenience wrapper. Use
    # function/tool calling via `chat(..., tools=[...])` exclusively.


def build_standard_metadata(
    *,
    app: str,
    component: str,
    function: str,
    conversation_id: Optional[str] = None,
    turn_index: Optional[int] = None,
    request_id: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Construct a standard metadata dict for AI calls.

    This shape is consistent across the codebase for attribution and logging.
    """
    md: Dict[str, Any] = {
        "app": app,
        "component": component,
        "function": function,
        "request_id": request_id or str(uuid.uuid4()),
    }
    if conversation_id is not None:
        md["conversation_id"] = conversation_id
    if turn_index is not None:
        md["turn_index"] = int(turn_index)
    if tags is not None:
        md["tags"] = list(tags)
    if extra:
        md.update(extra)
    return md
