"""Structured TRACE events for end-to-end request / CLI / provider logging.

Emitted lines are merged into JSON log rows by ``config.logging_config``.
Conversation and Claude Code prompts are logged verbatim unless values live under
sanitized credential keys (e.g. ``api_key``, ``authorization``).

A module-level ``_TRACING_ENABLED`` flag (controlled by env ``FCC_TRACING``)
lets operators disable trace overhead entirely in production.  When disabled,
``trace_event()`` returns immediately and ``traced_async_stream()`` skips
periodic chunk emissions while still forwarding stream data unchanged.
"""

import asyncio
import os
import sys
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from typing import Any

from loguru import logger

from free_claude_code.core.async_iterators import try_close_async_iterator

TRACE_PAYLOAD_BINDING = "trace_payload"

# Operators can disable all TRACE overhead via FCC_TRACING=false
_TRACING_ENABLED: bool = os.getenv("FCC_TRACING", "true").lower() not in (
    "0",
    "false",
    "no",
    "off",
)

_SECRET_VALUE_KEYS = frozenset(
    k.lower()
    for k in (
        "authorization",
        "x-api-key",
        "anthropic-auth-token",
        "api_key",
        "password",
        "secret",
        "token",
        "bearer_token",
        "openapi_token",
        "nvidia-api-key",
    )
)


def sanitize_trace_value(obj: Any) -> Any:
    """Recursively copy JSON-like structures redacting credential-shaped keys."""
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if str(k).lower() in _SECRET_VALUE_KEYS:
                out[str(k)] = "<redacted>"
            else:
                out[str(k)] = sanitize_trace_value(v)
        return out
    if isinstance(obj, tuple | list):
        return [sanitize_trace_value(x) for x in obj]
    return obj


def trace_event(*, stage: str, event: str, source: str, **fields: Any) -> None:
    """Emit one structured TRACE row (merged into JSON by the log sink).

    Returns immediately when tracing is disabled (``FCC_TRACING=false``).
    """
    if not _TRACING_ENABLED:
        return
    payload = sanitize_trace_value(
        {
            "stage": stage,
            "event": event,
            "source": source,
            **fields,
        },
    )
    logger.bind(trace_payload=payload).info("TRACE {}", event)


async def close_stream_input(
    iterator: object,
    *,
    owner: str,
    source: str,
    preserved_error: BaseException | None,
) -> None:
    """Close one transform input and observe cleanup failure without raising it."""
    close_error = await try_close_async_iterator(iterator)
    if close_error is None:
        return
    trace_event(
        stage="lifecycle",
        event="stream.input.close_failed",
        source=source,
        owner=owner,
        close_exc_type=type(close_error).__name__,
        preserved_exc_type=(
            type(preserved_error).__name__ if preserved_error is not None else None
        ),
    )


def extract_claude_session_id_from_headers(headers: Mapping[str, str]) -> str | None:
    """Best-effort session id forwarded by Claude Code / SDK via HTTP."""
    lowered = {str(k).lower(): v for k, v in headers.items() if isinstance(v, str)}
    for key in (
        "anthropic-session-id",
        "x-anthropic-session-id",
        "claude-session-id",
        "x-claude-session-id",
    ):
        candidate = lowered.get(key)
        if candidate:
            return candidate
    return None


async def traced_async_stream(
    agen: AsyncIterator[str],
    *,
    stage: str,
    source: str,
    complete_event: str,
    interrupted_event: str,
    chunk_event: str | None = None,
    chunk_interval: int = 250,
    extra: Mapping[str, Any] | None = None,
) -> AsyncGenerator[str]:
    """Emit TRACE rows when a text stream completes, fails, cancels, or periodically."""
    common = dict(extra or {})
    count = 0
    nbytes = 0
    interrupted = False
    try:
        async for chunk in agen:
            count += 1
            nbytes += len(chunk.encode("utf-8", errors="replace"))
            if (
                _TRACING_ENABLED
                and chunk_event
                and chunk_interval > 0
                and count % chunk_interval == 0
            ):
                trace_event(
                    stage=stage,
                    event=chunk_event,
                    source=source,
                    stream_chunks_so_far=count,
                    stream_bytes_so_far=nbytes,
                    **common,
                )
            yield chunk
    except GeneratorExit:
        raise
    except asyncio.CancelledError:
        interrupted = True
        trace_event(
            stage=stage,
            event=interrupted_event,
            source=source,
            stream_chunks=count,
            stream_bytes=nbytes,
            outcome="cancelled",
            **common,
        )
        raise
    except BaseExceptionGroup as grp:
        interrupted = True
        trace_event(
            stage=stage,
            event=interrupted_event,
            source=source,
            stream_chunks=count,
            stream_bytes=nbytes,
            outcome="exception_group",
            note=str(grp),
            **common,
        )
        raise
    except Exception as exc:
        interrupted = True
        trace_event(
            stage=stage,
            event=interrupted_event,
            source=source,
            stream_chunks=count,
            stream_bytes=nbytes,
            outcome="error",
            exc_type=type(exc).__name__,
            **common,
        )
        raise
    finally:
        await close_stream_input(
            agen,
            owner="traced_async_stream",
            source=source,
            preserved_error=sys.exception(),
        )

    if not interrupted:
        trace_event(
            stage=stage,
            event=complete_event,
            source=source,
            stream_chunks=count,
            stream_bytes=nbytes,
            outcome="ok",
            **common,
        )


def redact_message_text(text: str) -> str:
    return f"<text:len={len(text)}>"


def redact_block(item: Any) -> Any:
    if isinstance(item, str):
        return redact_message_text(item)
    if isinstance(item, Mapping):
        res = dict(item)
        t = res.get("type")
        if t == "text" and "text" in res:
            res["text"] = redact_message_text(res["text"])
        elif t == "thinking" and "thinking" in res:
            res["thinking"] = f"<thinking:len={len(res['thinking'])}>"
        elif t == "redacted_thinking" and "data" in res:
            res["data"] = f"<thinking_redacted:len={len(res['data'])}>"
        elif t == "image" and "source" in res:
            res["source"] = "<redacted_image>"
        elif t == "tool_result" and "content" in res:
            c = res["content"]
            if isinstance(c, str):
                res["content"] = redact_message_text(c)
            elif isinstance(c, tuple | list):
                res["content"] = [redact_block(x) for x in c]
            elif isinstance(c, Mapping):
                res["content"] = redact_block(c)
        elif "text" in res and isinstance(res["text"], str):
            res["text"] = redact_message_text(res["text"])
        return res
    if isinstance(item, tuple | list):
        return [redact_block(x) for x in item]
    return item


def redact_message_content(content: Any) -> Any:
    if isinstance(content, str):
        return redact_message_text(content)
    if isinstance(content, tuple | list):
        return [redact_block(x) for x in content]
    return redact_block(content)


def redact_system_prompt(system: Any) -> Any:
    if system is None:
        return None
    if isinstance(system, str):
        return redact_message_text(system)
    if isinstance(system, tuple | list):
        return [redact_block(x) for x in system]
    return redact_block(system)


def redact_messages_list(messages: Any) -> Any:
    if not isinstance(messages, tuple | list):
        return messages
    out = []
    for msg in messages:
        if isinstance(msg, Mapping):
            res = dict(msg)
            if "content" in res:
                res["content"] = redact_message_content(res["content"])
            if "reasoning_content" in res and isinstance(res["reasoning_content"], str):
                res["reasoning_content"] = redact_message_text(res["reasoning_content"])
            out.append(res)
        else:
            out.append(msg)
    return out


def provider_chat_body_snapshot(body: Mapping[str, Any]) -> dict[str, Any]:
    """Sanitized OpenAI-compat chat body subset for traces (conversation text redacted)."""
    keys = ("model", "messages", "tools", "tool_choice", "temperature", "max_tokens")
    snap = {k: body[k] for k in keys if k in body and body[k] is not None}
    if "messages" in snap:
        snap["messages"] = redact_messages_list(snap["messages"])
    return sanitize_trace_value(snap)
