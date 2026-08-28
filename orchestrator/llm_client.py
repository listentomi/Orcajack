"""LiteLLM wrapper for multi-model API compatibility.

Supports:
  - SiliconFlow / OpenAI-compatible: ``openai/<model>`` + api_base override
  - Local vLLM: ``openai/<served-name>`` + api_base http://localhost:PORT/v1
  - OpenRouter:    ``openrouter/<provider>/<model>`` (litellm auto-reads
    OPENROUTER_API_KEY from env; HTTP-Referer + X-Title injected automatically
    for attribution). Alternatively use ``openai/<model>`` with api_base
    https://openrouter.ai/api/v1 and OPENAI_API_KEY set to the OR key.
  - Yunwu (云雾 API):  ``yunwu/<model>`` is auto-rewritten to
    ``openai/<model>`` with api_base ``https://yunwu.ai/v1`` and api_key
    read from ``YUNWU_API_KEY`` (falls back to ``OPENAI_API_KEY``).
    Alternatively use ``openai/<model>`` with --api-base
    https://yunwu.ai/v1 and OPENAI_API_KEY set to the Yunwu key.

Set ``OPENROUTER_APP_NAME`` and ``OPENROUTER_REFERRER`` env vars to customize
the attribution headers. Both default to a neutral placeholder; set them to
your own project name / URL if you want OpenRouter attribution.
Override the Yunwu endpoint via ``YUNWU_API_BASE`` if needed.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any

import litellm

logger = logging.getLogger(__name__)

litellm.drop_params = True


def _is_openrouter_model(model: str) -> bool:
    return isinstance(model, str) and model.startswith("openrouter/")


def _openrouter_headers() -> dict[str, str]:
    """Per-OpenRouter-docs attribution headers."""
    return {
        "HTTP-Referer": os.environ.get("OPENROUTER_REFERRER", "https://localhost"),
        "X-Title": os.environ.get("OPENROUTER_APP_NAME", "orcajack"),
    }


_YUNWU_DEFAULT_API_BASE = "https://yunwu.ai/v1"


def _is_yunwu_model(model: str) -> bool:
    return isinstance(model, str) and model.startswith("yunwu/")


def _resolve_yunwu(model: str, api_base: str | None) -> tuple[str, str, str | None]:
    """Translate ``yunwu/<m>`` into a litellm-friendly ``openai/<m>`` call.

    Returns ``(rewritten_model, rewritten_api_base, yunwu_api_key_or_None)``.
    The api_key is read from ``YUNWU_API_KEY`` (preferred) or ``OPENAI_API_KEY``
    so users with a single env var still work.

    For ``yunwu/<m>`` prefix: ALWAYS forces the endpoint to ``https://yunwu.ai/v1``
    (or ``YUNWU_API_BASE`` if set), discarding any caller-supplied api_base —
    that param often leaks in from a stale config.yaml default. For Pattern A
    (``openai/<m>`` + caller supplied an explicit yunwu.ai api_base) the caller
    URL is honored as-is.
    """
    yunwu_default = os.environ.get("YUNWU_API_BASE", _YUNWU_DEFAULT_API_BASE)
    if _is_yunwu_model(model):
        rest = model.split("/", 1)[1]
        rewritten_model = f"openai/{rest}"
        rewritten_api_base = yunwu_default
    else:
        rewritten_model = model
        rewritten_api_base = api_base or yunwu_default
    api_key = os.environ.get("YUNWU_API_KEY") or os.environ.get("OPENAI_API_KEY")
    return rewritten_model, rewritten_api_base, api_key


@dataclass
class LLMUsage:
    """Token usage from a single LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""


class UsageAccumulator:
    """Thread-safe accumulator for token usage across multiple calls."""

    def __init__(self) -> None:
        self._calls: list[LLMUsage] = []
        self._lock = threading.Lock()

    def add(self, usage: LLMUsage) -> None:
        with self._lock:
            self._calls.append(usage)

    @property
    def calls(self) -> list[LLMUsage]:
        with self._lock:
            return list(self._calls)

    @property
    def total_prompt_tokens(self) -> int:
        with self._lock:
            return sum(u.prompt_tokens for u in self._calls)

    @property
    def total_completion_tokens(self) -> int:
        with self._lock:
            return sum(u.completion_tokens for u in self._calls)

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return sum(u.total_tokens for u in self._calls)

    @property
    def num_calls(self) -> int:
        with self._lock:
            return len(self._calls)

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()


def call_llm(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    response_format: dict[str, Any] | None = None,
    api_base: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> tuple[str, LLMUsage]:
    """Call an LLM via LiteLLM's unified interface.

    Returns (content, usage). Automatically injects OpenRouter attribution
    headers when ``model`` uses the ``openrouter/...`` prefix, or when the
    ``api_base`` points to ``openrouter.ai``.
    """
    # Yunwu auto-resolution: ``yunwu/<m>`` or api_base on yunwu.ai both flow
    # through the OpenAI-compatible relay; we just need to flip the model
    # prefix and inject the Yunwu key.
    using_yunwu = _is_yunwu_model(model) or (
        isinstance(api_base, str) and "yunwu.ai" in api_base
    )
    if using_yunwu:
        model, api_base, yunwu_key = _resolve_yunwu(model, api_base)
    else:
        yunwu_key = None

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    if api_base:
        kwargs["api_base"] = api_base
    if using_yunwu and yunwu_key:
        kwargs["api_key"] = yunwu_key

    # GLM-5+ reasoning models: passing thinking.type=disabled tells the model to
    # still reason (it can't be fully turned off) BUT to also emit the final
    # answer in `content`. Without it, the model blows past max_tokens in
    # reasoning_content and leaves content empty → JSON parse fails.
    model_lower = model.lower()
    if "glm-5" in model_lower:
        kwargs.setdefault("extra_body", {})
        kwargs["extra_body"]["thinking"] = {"type": "disabled"}

    # Auto-inject OpenRouter headers for attribution / rate-limit friendliness
    headers = dict(extra_headers) if extra_headers else {}
    using_openrouter = _is_openrouter_model(model) or (
        isinstance(api_base, str) and "openrouter.ai" in api_base
    )
    if using_openrouter:
        for k, v in _openrouter_headers().items():
            headers.setdefault(k, v)
    if headers:
        kwargs["extra_headers"] = headers

    logger.info("Calling model=%s  tokens=%d  api_base=%s", model, max_tokens, api_base or "default")
    response = litellm.completion(**kwargs)
    content = response.choices[0].message.content
    logger.debug("Raw LLM response: %s", content[:200])

    usage = LLMUsage(model=model)
    if response.usage:
        usage.prompt_tokens = response.usage.prompt_tokens or 0
        usage.completion_tokens = response.usage.completion_tokens or 0
        usage.total_tokens = response.usage.total_tokens or 0
    logger.info(
        "Token usage: prompt=%d completion=%d total=%d",
        usage.prompt_tokens, usage.completion_tokens, usage.total_tokens,
    )
    return content, usage


def call_llm_json(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    api_base: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> tuple[dict, LLMUsage]:
    """Call LLM and parse the response as JSON.

    Returns (parsed_dict, usage).
    """
    try:
        raw, usage = call_llm(
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            api_base=api_base,
            extra_headers=extra_headers,
        )
    except Exception:
        logger.warning("JSON mode not supported for %s, falling back to plain text", model)
        raw, usage = call_llm(
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=api_base,
            extra_headers=extra_headers,
        )

    return _parse_json(raw), usage


def _parse_json(text: str) -> dict:
    """Best-effort JSON extraction from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)
