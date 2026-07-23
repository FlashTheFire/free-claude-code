"""Structured TRACE logging assertions."""

import json
from pathlib import Path

import pytest
from loguru import logger

from free_claude_code.config.logging_config import configure_logging
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.core.trace import (
    TRACE_PAYLOAD_BINDING,
    trace_event,
    traced_async_stream,
)


class _CloseTrackingIterator:
    def __init__(
        self,
        chunks: list[str],
        *,
        iteration_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self._chunks = iter(chunks)
        self._iteration_error = iteration_error
        self._close_error = close_error
        self.close_calls = 0

    def __aiter__(self) -> _CloseTrackingIterator:
        return self

    async def __anext__(self) -> str:
        try:
            return next(self._chunks)
        except StopIteration:
            if self._iteration_error is not None:
                error = self._iteration_error
                self._iteration_error = None
                raise error from None
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        self.close_calls += 1
        if self._close_error is not None:
            raise self._close_error


def _json_log_rows(log_file: str) -> list[dict]:
    logger.complete()
    text = Path(log_file).read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [json.loads(line) for line in text.split("\n")]


def test_trace_payload_merged_into_json_line(tmp_path) -> None:
    log_file = str(tmp_path / "t.log")
    configure_logging(log_file, force=True)
    trace_event(stage="s", event="e.v1", source="unit", hello="world", n=42)
    row = _json_log_rows(log_file)[-1]
    assert row["trace"] is True
    assert row["stage"] == "s"
    assert row["event"] == "e.v1"
    assert row["source"] == "unit"
    assert row["hello"] == "world"
    assert row["n"] == 42
    assert TRACE_PAYLOAD_BINDING == "trace_payload"


def test_sanitize_masks_nested_api_key_strings() -> None:
    """Credential-shaped keys redact without touching normal message text."""
    from free_claude_code.core.trace import sanitize_trace_value

    out = sanitize_trace_value(
        {"outer": {"api_key": "secret", "text": "visible"}},
    )
    assert out["outer"]["api_key"] == "<redacted>"
    assert out["outer"]["text"] == "visible"


@pytest.mark.asyncio
async def test_traced_async_stream_logs_completion(tmp_path) -> None:
    log_file = str(tmp_path / "complete.log")
    configure_logging(log_file, force=True)

    source = _CloseTrackingIterator(["hello", " world"])

    chunks = [
        chunk
        async for chunk in traced_async_stream(
            source,
            stage="egress",
            source="unit",
            complete_event="stream.completed",
            interrupted_event="stream.interrupted",
            extra={"request_id": "req_complete"},
        )
    ]

    assert chunks == ["hello", " world"]
    assert source.close_calls == 1
    rows = _json_log_rows(log_file)
    completed = [row for row in rows if row.get("event") == "stream.completed"]
    assert len(completed) == 1
    assert completed[0]["request_id"] == "req_complete"
    assert completed[0]["stream_chunks"] == 2
    assert completed[0]["outcome"] == "ok"


@pytest.mark.asyncio
async def test_traced_async_stream_logs_real_exception(tmp_path) -> None:
    log_file = str(tmp_path / "error.log")
    configure_logging(log_file, force=True)

    source = _CloseTrackingIterator(
        ["before"],
        iteration_error=RuntimeError("boom"),
        close_error=RuntimeError("close boom"),
    )

    with pytest.raises(RuntimeError, match="boom"):
        async for _chunk in traced_async_stream(
            source,
            stage="egress",
            source="unit",
            complete_event="stream.completed",
            interrupted_event="stream.interrupted",
            extra={"request_id": "req_error"},
        ):
            pass

    assert source.close_calls == 1

    rows = _json_log_rows(log_file)
    interrupted = [row for row in rows if row.get("event") == "stream.interrupted"]
    assert len(interrupted) == 1
    assert interrupted[0]["request_id"] == "req_error"
    assert interrupted[0]["stream_chunks"] == 1
    assert interrupted[0]["outcome"] == "error"
    assert interrupted[0]["exc_type"] == "RuntimeError"
    close_failed = [
        row for row in rows if row.get("event") == "stream.input.close_failed"
    ]
    assert len(close_failed) == 1
    assert close_failed[0]["owner"] == "traced_async_stream"
    assert close_failed[0]["close_exc_type"] == "RuntimeError"
    assert close_failed[0]["preserved_exc_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_traced_async_stream_closes_quietly_on_generator_exit(tmp_path) -> None:
    log_file = str(tmp_path / "generator_exit.log")
    configure_logging(log_file, force=True)

    source = _CloseTrackingIterator(["first", "second"])

    stream = traced_async_stream(
        source,
        stage="egress",
        source="unit",
        complete_event="stream.completed",
        interrupted_event="stream.interrupted",
        extra={"request_id": "req_closed"},
    )

    assert await anext(stream) == "first"
    await stream.aclose()

    assert source.close_calls == 1
    rows = _json_log_rows(log_file)
    events = {row.get("event") for row in rows}
    assert "stream.completed" not in events
    assert "stream.interrupted" not in events


def test_prompt_redaction() -> None:
    """Verifies that message and system text content are redacted from traces."""
    from free_claude_code.core.anthropic.request_snapshot import (
        anthropic_request_snapshot,
    )
    from free_claude_code.core.trace import provider_chat_body_snapshot

    # Test OpenAI/compat compat snapshot
    body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
            {
                "role": "assistant",
                "content": "I am fine.",
                "reasoning_content": "Internal thought process",
            },
        ],
        "temperature": 0.7,
    }
    snap = provider_chat_body_snapshot(body)
    assert snap["model"] == "gpt-4o"
    assert snap["temperature"] == 0.7
    assert snap["messages"][0]["role"] == "user"
    assert snap["messages"][0]["content"] == "<text:len=19>"
    assert snap["messages"][1]["role"] == "assistant"
    assert snap["messages"][1]["content"] == "<text:len=10>"
    assert snap["messages"][1]["reasoning_content"] == "<text:len=24>"

    # Test Anthropic messages request snapshot
    req = MessagesRequest(
        model="claude-3-5-sonnet",
        messages=[
            Message(role="user", content="What is the capital of France?"),
        ],
        system="Act as a helpful assistant.",
        max_tokens=1024,
    )
    snap2 = anthropic_request_snapshot(req)
    assert snap2["model"] == "claude-3-5-sonnet"
    assert snap2["max_tokens"] == 1024
    assert snap2["system"] == "<text:len=27>"
    assert snap2["messages"][0]["role"] == "user"
    assert snap2["messages"][0]["content"] == "<text:len=30>"


def test_console_formatter() -> None:
    """Verifies that _console_formatter formats different log types correctly."""
    from free_claude_code.config.logging_config import _console_formatter
    from free_claude_code.core.trace import TRACE_PAYLOAD_BINDING

    class MockRecord(dict):
        def __getattr__(self, name):
            return self[name]

    # Test REQUEST_COMPLETE
    rec1 = MockRecord(
        {
            "time": pytest.importorskip("datetime").datetime(2026, 7, 12, 22, 28, 17),
            "level": MockRecord({"name": "INFO"}),
            "message": "REQUEST_COMPLETE method=POST path=/v1/messages status=200 elapsed_ms=31.1",
            "name": "free_claude_code.api",
            "line": 42,
        }
    )
    res1 = _console_formatter(rec1)
    assert "REQ_OK" in res1
    assert "POST" in res1
    assert "status=<green>200" in res1
    assert "elapsed=<green>31.1ms" in res1

    # Test TRACE event
    rec2 = MockRecord(
        {
            "time": pytest.importorskip("datetime").datetime(2026, 7, 12, 22, 28, 17),
            "level": MockRecord({"name": "INFO"}),
            "message": "TRACE provider.request.sent",
            "name": "free_claude_code.core",
            "line": 50,
            "extra": {
                TRACE_PAYLOAD_BINDING: {
                    "event": "provider.request.sent",
                    "stage": "provider",
                    "provider_id": "gemini",
                    "model": "gemini-2.5-pro",
                }
            },
        }
    )
    res2 = _console_formatter(rec2)
    assert "TRACE" in res2
    assert "provider.request.sent" in res2
    assert "provider=<yellow>gemini" in res2
    assert "model=<cyan>gemini-2.5-pro" in res2

    # Test Discovery
    rec3 = MockRecord(
        {
            "time": pytest.importorskip("datetime").datetime(2026, 7, 12, 22, 28, 17),
            "level": MockRecord({"name": "INFO"}),
            "message": "Provider model discovery cached: provider=cerebras models=3",
            "name": "free_claude_code.providers",
            "line": 15,
        }
    )
    res3 = _console_formatter(rec3)
    assert "DISCOVER" in res3
    assert "cerebras" in res3
    assert "3" in res3

    # Test Fallback standard log
    rec4 = MockRecord(
        {
            "time": pytest.importorskip("datetime").datetime(2026, 7, 12, 22, 28, 17),
            "level": MockRecord({"name": "WARNING"}),
            "message": "Hello {world}",
            "name": "free_claude_code.runtime",
            "line": 110,
        }
    )
    res4 = _console_formatter(rec4)
    assert "WARNING" in res4
    assert "runtime:110" in res4
    assert "Hello {{world}}" in res4
