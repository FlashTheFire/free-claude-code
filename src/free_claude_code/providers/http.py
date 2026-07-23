"""Shared HTTP lifecycle helpers for upstream provider clients."""

import inspect
import time
from typing import Any

from loguru import logger

from free_claude_code.core.trace import trace_event


async def maybe_await_aclose(response: Any) -> None:
    """Call ``aclose`` on httpx-like responses; ignore sync test doubles."""
    close = getattr(response, "aclose", None)
    if not callable(close):
        return
    result = close()
    if inspect.isawaitable(result):
        await result


async def close_provider_stream(
    stream: Any,
    *,
    active_error: BaseException | None,
    provider_name: str,
    request_id: str | None,
) -> None:
    """Close one stream without letting cleanup change its established outcome."""
    t0 = time.monotonic()
    try:
        await maybe_await_aclose(stream)
    except Exception as close_error:
        active_error_type = (
            type(active_error).__name__ if active_error is not None else None
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        trace_event(
            stage="provider",
            event="provider.stream.close_failed",
            source="provider",
            provider=provider_name,
            request_id=request_id,
            close_exc_type=type(close_error).__name__,
            preserved_exc_type=active_error_type,
        )
        logger.warning(
            "{}_STREAM_CLOSE_FAILED request_id={} close_exc_type={} "
            "preserved_exc_type={} elapsed_ms={:.1f}",
            provider_name,
            request_id,
            type(close_error).__name__,
            active_error_type,
            elapsed_ms,
        )
    else:
        elapsed_ms = (time.monotonic() - t0) * 1000
        if elapsed_ms > 500:
            logger.info(
                "{}_STREAM_CLOSE_SLOW request_id={} elapsed_ms={:.1f}",
                provider_name,
                request_id,
                elapsed_ms,
            )
