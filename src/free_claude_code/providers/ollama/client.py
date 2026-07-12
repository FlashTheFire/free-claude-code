"""Ollama provider implementation."""

from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.defaults import OLLAMA_DEFAULT_BASE
from free_claude_code.providers.rate_limit import ProviderRateLimiter
from free_claude_code.providers.transports.openai_chat import (
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    build_openai_chat_request_body,
)

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="OLLAMA",
    default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
)


class OllamaProvider(OpenAIChatTransport):
    """Ollama provider using OpenAI-compatible Chat Completions."""

    def __init__(self, config: ProviderConfig, *, rate_limiter: ProviderRateLimiter):
        super().__init__(
            config,
            provider_name="OLLAMA",
            base_url=_openai_base_url(config.base_url or OLLAMA_DEFAULT_BASE),
            api_key=config.api_key or "ollama",
            rate_limiter=rate_limiter,
        )

    def _build_request_body(
        self, request: MessagesRequest, thinking_enabled: bool | None = None
    ) -> dict:
        return build_openai_chat_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
            policy=_REQUEST_POLICY,
        )


def _openai_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    return normalized if normalized.endswith("/v1") else f"{normalized}/v1"
