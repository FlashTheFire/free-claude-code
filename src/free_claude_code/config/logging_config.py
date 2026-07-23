"""Loguru-based structured logging configuration.

Structured logs are written as JSON lines to a configurable path (default
``logs/server.log``). Stdlib logging is intercepted and funneled to loguru.
Context vars (request_id, node_id, chat_id) from contextualize() are
included at top level for easy grep/filter.

Performance notes
-----------------
* The file sink defaults to **INFO** (configurable via ``log_level``).
  DEBUG-level noise from httpx/httpcore/asyncio is dropped before
  serialization, eliminating ~60 % of log volume in typical workloads.
* Serialization uses ``orjson`` when available (3-10x faster than stdlib
  ``json``); falls back transparently to ``json.dumps``.
"""

import contextlib
import importlib
import json
import logging
import re
import sys
import threading
from pathlib import Path
from typing import Any

from loguru import logger

# --- Fast JSON encoder (orjson when available) --------------------------------

_orjson: Any = None
with contextlib.suppress(ModuleNotFoundError):
    _orjson = importlib.import_module("orjson")


def _fast_json_dumps(obj: dict) -> str:
    if _orjson is not None:
        return _orjson.dumps(obj, default=str).decode("utf-8")
    return json.dumps(obj, default=str)


_configured = False

# Loguru ``logger.bind()`` key used by structured TRACE payloads; ``core/trace.py``
# uses the identical string constant ``TRACE_PAYLOAD_BINDING``.
_TRACE_PAYLOAD_BINDING = "trace_payload"

# Context keys we promote to top-level JSON for traceability / grep
_CONTEXT_KEYS = (
    "request_id",
    "node_id",
    "chat_id",
    "claude_session_id",
    "http_method",
    "http_path",
)

_TELEGRAM_BOT_RE = re.compile(
    r"(https?://api\.telegram\.org/)bot([0-9]+:[A-Za-z0-9_-]+)(/?)",
    re.IGNORECASE,
)
# Authorization: Bearer <token> (HTTP client / proxy debug lines)
_AUTH_BEARER_RE = re.compile(
    r"(\bAuthorization\s*:\s*Bearer\s+)([^\s'\"]+)",
    re.IGNORECASE,
)


def _redact_sensitive_substrings(message: str) -> str:
    """Remove obvious API tokens and secrets before JSON log line emission."""
    text = _TELEGRAM_BOT_RE.sub(r"\1bot<redacted>\3", message)
    return _AUTH_BEARER_RE.sub(r"\1<redacted>", text)


def _serialize_with_context(record) -> str:
    """Format record as JSON with context vars at top level.
    Returns a format template; we inject _json into record for output.
    """
    extra = record.get("extra", {})
    out = {
        "time": str(record["time"]),
        "level": record["level"].name,
        "message": _redact_sensitive_substrings(str(record["message"])),
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
    }
    trace_payload = extra.get(_TRACE_PAYLOAD_BINDING)
    for key in _CONTEXT_KEYS:
        if key in extra and extra[key] is not None:
            out[key] = extra[key]
    if isinstance(trace_payload, dict):
        for tk, tv in trace_payload.items():
            if tk in out:
                continue
            out[tk] = tv
        out["trace"] = True
    record["_json"] = _fast_json_dumps(out)
    return "{_json}\n"


def _console_formatter(record) -> str:
    """Format records for colored, human-readable terminal display."""
    extra = record.get("extra", {})
    msg = record["message"]

    # Escape curly braces in the message to prevent loguru formatting errors
    msg_escaped = str(msg).replace("{", "{{").replace("}", "}}")

    # 1. Specialized handling for Incoming requests
    if msg.startswith("Incoming "):
        return (
            f"\n<cyan>{{time:HH:mm:ss}}</cyan> | 📥 <b>INCOMING</b> | {msg_escaped}\n"
        )

    # 2. Specialized handling for REQUEST_COMPLETE
    if msg.startswith("REQUEST_COMPLETE "):
        parts = {}
        for part in msg.split():
            if "=" in part:
                k, v = part.split("=", 1)
                parts[k] = v
        method = parts.get("method", "")
        path = parts.get("path", "")
        status = parts.get("status", "")
        elapsed = parts.get("elapsed_ms", "")

        status_color = "green" if status.startswith("2") else "red"
        icon = "🟢" if status.startswith("2") else "🔴"

        try:
            val = float(elapsed)
            if val > 2000:
                elapsed_color = "red"
            elif val > 500:
                elapsed_color = "yellow"
            else:
                elapsed_color = "green"
        except Exception:
            elapsed_color = "green"

        return (
            f"<green>{{time:HH:mm:ss}}</green> | {icon} <b><green>REQ_OK</green></b>  | "
            f"<b>{method}</b> {path} | "
            f"status=<{status_color}>{status}</{status_color}> | "
            f"elapsed=<{elapsed_color}>{elapsed}ms</{elapsed_color}>\n"
        )

    # 3. Specialized handling for TRACE events
    trace_payload = extra.get(_TRACE_PAYLOAD_BINDING)
    if trace_payload and isinstance(trace_payload, dict):
        event = trace_payload.get("event", "")
        stage = trace_payload.get("stage", "")
        model = trace_payload.get("model") or trace_payload.get("provider_model")
        provider = trace_payload.get("provider_id") or trace_payload.get("provider")
        attempt = trace_payload.get("attempt")
        max_attempts = trace_payload.get("max_attempts")
        delay_s = trace_payload.get("delay_s")

        details = []
        if stage:
            details.append(f"stage=<magenta>{stage}</magenta>")
        if provider:
            details.append(f"provider=<yellow>{provider}</yellow>")
        if model:
            details.append(f"model=<cyan>{model}</cyan>")
        if attempt:
            details.append(f"attempt=<red>{attempt}/{max_attempts}</red>")
        if delay_s:
            details.append(f"delay=<yellow>{delay_s}s</yellow>")

        details_str = " | ".join(details)
        prefix = f" | {details_str}" if details else ""
        return (
            f"<cyan>{{time:HH:mm:ss}}</cyan> | ✦ <b>TRACE</b>   | "
            f"<magenta>{event}</magenta>{prefix}\n"
        )

    # 4. Model discovery cached
    if "Provider model discovery cached" in msg:
        m = re.search(r"provider=(\S+)\s+models=(\d+)", msg)
        if m:
            prov = m.group(1)
            cnt = m.group(2)
            return (
                f"<green>{{time:HH:mm:ss}}</green> | 🔎 <b>DISCOVER</b> | "
                f"Provider <yellow>{prov}</yellow> cached <green>{cnt}</green> models\n"
            )

    if "Provider model discovery skipped" in msg:
        m = re.search(r"provider=(\S+)\s+reason=(.+)", msg)
        if m:
            prov = m.group(1)
            reason = m.group(2).replace("{", "{{").replace("}", "}}")
            return (
                f"<green>{{time:HH:mm:ss}}</green> | 🔎 <b>DISCOVER</b> | "
                f"Provider <yellow>{prov}</yellow> <red>skipped</red> (reason: <red>{reason}</red>)\n"
            )

    # 5. Standard logs fallback
    level = record["level"].name
    if level == "INFO":
        lvl_color = "cyan"
        icon = "ℹ️"  # noqa: RUF001
    elif level == "WARNING":
        lvl_color = "yellow"
        icon = "⚠️"
    elif level == "ERROR":
        lvl_color = "red"
        icon = "🚨"
    elif level == "DEBUG":
        lvl_color = "blue"
        icon = "🪲"
    else:
        lvl_color = "magenta"
        icon = "⚙️"

    mod_name = record["name"].split(".")[-1]
    return (
        f"{{time:HH:mm:ss}} | {icon} <{lvl_color}><b>{level:<7}</b></{lvl_color}> | "
        f"<blue>{mod_name}:{record['line']}</blue> | "
        f"{msg_escaped}\n"
    )


class InterceptHandler(logging.Handler):
    """Redirect stdlib logging to loguru."""

    __slots__ = ("_local",)

    def __init__(self) -> None:
        super().__init__()
        self._local = threading.local()

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(self._local, "active", False):
            # Avoid deadlock when nested stdlib records fire during a loguru emit.
            return
        self._local.active = True
        try:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            frame, depth = logging.currentframe(), 2
            while frame is not None and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
        finally:
            self._local.active = False


def configure_logging(
    log_file: str | Path,
    *,
    force: bool = False,
    verbose_third_party: bool = False,
    log_level: str = "INFO",
) -> None:
    """Configure loguru with JSON output to log_file and intercept stdlib logging.

    Idempotent: skips if already configured (e.g. hot reload).
    Use force=True to reconfigure (e.g. in tests with a different log path).

    When ``verbose_third_party`` is false, noisy HTTP and Telegram loggers are capped
    at WARNING unless explicitly configured otherwise.

    ``log_level`` controls the file sink's minimum level (default ``INFO``).
    Set to ``DEBUG`` for verbose troubleshooting.
    """
    global _configured
    if _configured and not force:
        return
    _configured = True

    # Remove default loguru handler (writes to stderr)
    logger.remove()

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Truncate log file on fresh start for clean debugging
    log_path.write_text("")

    # Validate and normalise the requested level
    effective_level = log_level.upper() if log_level else "INFO"
    if effective_level not in ("TRACE", "DEBUG", "INFO", "WARNING", "ERROR"):
        effective_level = "INFO"

    logger.add(
        log_file,
        level=effective_level,
        format=_serialize_with_context,
        encoding="utf-8",
        mode="a",
        rotation="50 MB",
        enqueue=True,
    )

    # Add colored console output to stderr
    logger.add(
        sys.stderr,
        level=effective_level,
        format=_console_formatter,
        colorize=True,
        enqueue=True,
    )

    # Intercept stdlib logging: route all root logger output to loguru
    intercept = InterceptHandler()
    logging.root.handlers = [intercept]
    logging.root.setLevel(logging.DEBUG)

    third_party = (
        "httpx",
        "httpcore",
        "httpcore.http11",
        "httpcore.connection",
        "telegram",
        "telegram.ext",
    )
    for name in third_party:
        logging.getLogger(name).setLevel(
            logging.WARNING if not verbose_third_party else logging.NOTSET
        )
