"""Small tracing setup adapted from Unit 4 / Lesson 10 demos 02-05.

Default: print a safe span tree locally.
Optional: export auto-instrumented Agent traces to Langfuse when the application is
started with ``--langfuse`` and Langfuse credentials are configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import os
from typing import Any

from agents import set_trace_processors, set_tracing_disabled
from agents.tracing import TracingProcessor


def _duration_ms(span: Any) -> float | None:
    """Return elapsed milliseconds for a completed span."""

    if not span.started_at or not span.ended_at:
        return None
    started = datetime.fromisoformat(span.started_at)
    ended = datetime.fromisoformat(span.ended_at)
    return (ended - started).total_seconds() * 1000


class ConsoleTraceProcessor(TracingProcessor):
    """Print span names, latency, and errors without prompt or tool payloads."""

    def __init__(self) -> None:
        self._spans: list[Any] = []

    def on_trace_start(self, trace) -> None:
        print(f"\nTRACE | {trace.name}")

    def on_trace_end(self, _trace) -> None:
        children: dict[str | None, list[Any]] = {}
        for span in self._spans:
            children.setdefault(span.parent_id, []).append(span)

        def walk(parent_id: str | None, depth: int = 0) -> None:
            for span in children.get(parent_id, []):
                data = span.span_data
                label = getattr(data, "name", None) or data.type
                elapsed = _duration_ms(span)
                latency = f"{elapsed:.1f} ms" if elapsed is not None else "? ms"
                status = "ERROR" if span.error else "OK"
                print(f"{'  ' * depth}└─ {label} | {status} | {latency}")
                walk(span.span_id, depth + 1)

        walk(None)
        self._spans.clear()

    def on_span_start(self, _span) -> None:
        pass

    def on_span_end(self, span) -> None:
        self._spans.append(span)

    def force_flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


@dataclass
class Observability:
    """Runtime observability state and optional Langfuse client."""

    mode: str
    langfuse_client: Any | None = None

    def flush(self) -> None:
        if self.langfuse_client is not None:
            self.langfuse_client.flush()


def trace_group_id(session_id: str) -> str:
    """Create a stable, non-reversible session label for trace grouping."""

    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]
    return f"payment-planner-{digest}"


def configure_observability(use_langfuse: bool = False) -> Observability:
    """Enable local traces and optionally add Langfuse auto-instrumentation."""

    # Replaces the SDK's default OpenAI trace uploader. Local mode therefore stays
    # local even when OPENAI_API_KEY is configured.
    set_trace_processors([ConsoleTraceProcessor()])
    set_tracing_disabled(False)

    if not use_langfuse:
        print("Observability: local span tree enabled; no Langfuse export.")
        return Observability(mode="local")

    required = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
    missing = [name for name in required if not os.getenv(name, "").strip()]
    if missing:
        raise RuntimeError(
            "Langfuse export requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY."
        )
    if not os.getenv("LANGFUSE_BASE_URL") and not os.getenv("LANGFUSE_HOST"):
        os.environ["LANGFUSE_BASE_URL"] = "http://localhost:3000"

    try:
        from langfuse import get_client
        from openinference.instrumentation import TraceConfig
        from openinference.instrumentation.openai_agents import (
            OpenAIAgentsInstrumentor,
        )
    except ImportError as error:
        raise RuntimeError(
            "Install the Payment Planner requirements before using --langfuse."
        ) from error

    # Financial prompts and tool payloads can be private. Keep the useful execution
    # structure, timings, errors, and usage while hiding message content.
    privacy_config = TraceConfig(
        hide_inputs=True,
        hide_outputs=True,
        hide_input_messages=True,
        hide_output_messages=True,
    )
    # Keep the console processor and add Langfuse's OpenTelemetry processor beside it.
    OpenAIAgentsInstrumentor().instrument(
        config=privacy_config,
        exclusive_processor=False,
    )
    client = get_client()
    print("Observability: local span tree + Langfuse export enabled.")
    return Observability(mode="langfuse", langfuse_client=client)


def run_local_demo() -> None:
    """Create an offline trace proving latency and error display work."""

    from agents import custom_span, function_span, trace

    configure_observability()
    with trace("payment_planner_demo"):
        with custom_span("memory_retrieval", data={"recalled_count": 2}):
            pass
        with function_span(name="categorize_transaction"):
            pass
        with custom_span("token_usage", data={"total_tokens": 120}):
            pass
        with custom_span("sample_error") as span:
            span.set_error(
                {"message": "DemonstrationError", "data": {"stage": "demo"}}
            )


if __name__ == "__main__":
    run_local_demo()
