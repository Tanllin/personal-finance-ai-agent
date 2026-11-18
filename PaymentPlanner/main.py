"""Command-line entry point for the standalone Payment Planner."""

from __future__ import annotations

import argparse
import asyncio
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from agent import build_agent
from long_term_memory import (
    LongTermPreferenceMemory,
    contains_sensitive_input,
    should_remember,
)
from memory import (
    RUN_CONFIG,
    IdempotencyStore,
    KEEP_ITEMS,
    count_items_tokens,
    create_compaction_session,
    create_session,
    idempotency_demo,
    keep_recent_history,
    load_planner_context,
    memory_demo,
    planner_profile_summary,
)
from observability import Observability, configure_observability, trace_group_id
from tools import categorize_transaction, parse_transaction


@dataclass(frozen=True)
class BackendSettings:
    supports_compaction: bool
    compaction_model: str


def configure_model_backend() -> BackendSettings:
    """Use OpenAI when configured, otherwise use a running local Ollama."""

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key and api_key != "ollama":
        return BackendSettings(
            supports_compaction=True,
            compaction_model=os.getenv("COMPACTION_MODEL", "gpt-4.1"),
        )

    base_url = os.getenv("LOCAL_BASE_URL", "http://localhost:11434/v1")
    model = os.getenv("LOCAL_MODEL", "qwen2.5:7b")
    try:
        urllib.request.urlopen(
            base_url.removesuffix("/v1") + "/api/version",
            timeout=0.8,
        )
    except Exception as error:
        raise RuntimeError(
            "Set OPENAI_API_KEY in .env or start Ollama before chat mode."
        ) from error

    from openai import AsyncOpenAI
    import agents
    from agents import (
        set_default_openai_api,
        set_default_openai_client,
        set_tracing_disabled,
    )

    os.environ["OPENAI_API_KEY"] = "ollama"
    set_default_openai_client(AsyncOpenAI(base_url=base_url, api_key="ollama"))
    set_default_openai_api("chat_completions")
    set_tracing_disabled(True)

    original_init = agents.Agent.__init__

    def init_with_local_model(self, *args, **kwargs):
        if not kwargs.get("model"):
            kwargs["model"] = model
        original_init(self, *args, **kwargs)

    agents.Agent.__init__ = init_with_local_model
    print(f"Using local Ollama model: {model}")
    return BackendSettings(supports_compaction=False, compaction_model="")


def run_tool_selftest() -> None:
    """Offline tests for transaction tools, P3 routing, and topic replacement."""

    cases = [
        ("UBER *TRIP $20.00", "Transportation"),
        ("2026-08-30 UBER EATS $18.25", "Dining"),
        ("WHOLE FOODS 12.50 USD", "Groceries"),
        ("LOCAL BOOK SHOP $23.75", "Other"),
    ]
    for raw_input, expected_category in cases:
        parsed = parse_transaction(raw_input)
        categorized = categorize_transaction(
            parsed["description"] or "",
            parsed["merchant"],
        )
        assert parsed["status"] == "ok"
        assert categorized["category"] == expected_category
        print(f"[PASS] {raw_input} -> {expected_category}")

    rejected = parse_transaction("card 4111 1111 1111 1111 $20.00")
    assert rejected["status"] == "error"
    assert rejected["raw_input"] == "[redacted]"
    print("[PASS] Likely banking credentials are rejected and redacted")

    assert should_remember("My monthly budget is 3000") == (
        True,
        "constraint",
        "monthly_budget",
    )
    assert should_remember("UBER *TRIP $20.00")[0] is False
    assert should_remember("My PIN is 1234")[0] is False
    assert should_remember("Categorize CORNER SHOP as Groceries") == (
        True,
        "category_preference",
        "merchant_category:corner_shop",
    )
    print("[PASS] P3 routing keeps preferences but excludes transactions and credentials")

    test_memory = LongTermPreferenceMemory(
        user_id="payment_planner_selftest",
        collection="payment-planner-p3-selftest-v1",
    )
    test_memory.forget("monthly_budget")
    test_memory.remember(
        "My monthly budget is 3000",
        kind="constraint",
        topic="monthly_budget",
    )
    test_memory.remember(
        "Actually my monthly budget is 2500",
        kind="constraint",
        topic="monthly_budget",
    )
    assert test_memory.count_topic("monthly_budget") == 1
    assert "2500" in (test_memory.peek_topic("monthly_budget") or "")
    test_memory.forget("monthly_budget")
    print("[PASS] Same-topic corrections replace the previous long-term memory")
    print("Payment Planner tool self-test passed.")


async def run_chat(
    session_id: str,
    backend: BackendSettings,
    observability: Observability,
    reset_session: bool = False,
) -> None:
    """Run the interactive Agent with SQLite history and structured context."""

    from agents import Runner, custom_span, trace

    context = load_planner_context()
    user_id = os.getenv("PAYMENT_PLANNER_USER_ID", "payment_planner_default")
    long_term_memory = LongTermPreferenceMemory(user_id=user_id)
    underlying_session = create_session(session_id)
    compaction_session = (
        create_compaction_session(
            session_id,
            underlying_session,
            backend.compaction_model,
        )
        if backend.supports_compaction
        else None
    )
    session = compaction_session or underlying_session
    idempotency = IdempotencyStore()
    if reset_session:
        await session.clear_session()
        idempotency.clear_session(session_id)
        print(f"Cleared conversation history for: {session_id}")

    print("Payment Planner started. Type exit to stop.")
    print(f"Session: {session_id}")
    print(f"Sliding context window: latest {KEEP_ITEMS} stored items")
    print(
        "Compaction: "
        + ("enabled for OpenAI Responses" if compaction_session else "disabled for this backend")
    )
    print(planner_profile_summary(context))
    print("Try: UBER *TRIP $20.00")
    print(f"Long-term user scope: {user_id}; memories: {long_term_memory.count()}")
    print("Commands: /profile, /memory, /long-memory, /forget TOPIC, /compact, /reset, exit")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in {"exit", "quit"} or user_input == "退出":
            break
        if not user_input:
            continue
        if contains_sensitive_input(user_input):
            print(
                "Payment Planner: sensitive banking credentials are not accepted. "
                "Remove them and resend only the transaction descriptor."
            )
            continue
        if user_input == "/profile":
            print(planner_profile_summary(context))
            continue
        if user_input == "/memory":
            items = await underlying_session.get_items()
            window = keep_recent_history(items, [])
            print(
                f"Stored session items: {len(items)}; "
                f"recent history sent per run: up to {KEEP_ITEMS} items."
            )
            print(
                "tiktoken estimate — "
                f"stored history: {count_items_tokens(items)} tokens; "
                f"current window: {count_items_tokens(window)} tokens."
            )
            continue
        if user_input == "/long-memory":
            print(f"Stored long-term preferences for this user: {long_term_memory.count()}")
            continue
        if user_input.startswith("/forget "):
            topic = user_input.removeprefix("/forget ").strip()
            removed = long_term_memory.forget(topic)
            print(f"Removed {removed} long-term memories for topic: {topic}")
            continue
        if user_input == "/compact":
            if compaction_session is None:
                print("Compaction requires an OpenAI Responses-compatible backend.")
                continue
            before = len(await underlying_session.get_items())
            await compaction_session.run_compaction({"force": True})
            after = len(await underlying_session.get_items())
            print(f"Compaction complete: {before} items -> {after} summarized items.")
            continue
        if user_input == "/reset":
            await session.clear_session()
            idempotency.clear_session(session_id)
            print("Conversation history cleared. Structured profile retained.")
            continue

        status, digest, cached_response = idempotency.begin(session_id, user_input)
        if status == "duplicate":
            print("Payment Planner [idempotent cached response]:", cached_response)
            continue
        if status == "pending":
            print("Payment Planner: this identical input is already being processed.")
            continue

        history_before = await underlying_session.get_items()
        model_input_estimate = keep_recent_history(
            history_before,
            [{"role": "user", "content": user_input}],
        )
        prompt_token_estimate = count_items_tokens(model_input_estimate)
        try:
            with trace(
                "payment_planner_run",
                group_id=trace_group_id(session_id),
                metadata={
                    "app": "payment_planner",
                    "observability_mode": observability.mode,
                },
            ):
                try:
                    with custom_span(
                        "memory_retrieval",
                        data={"query_character_count": len(user_input)},
                    ) as memory_span:
                        recalled = long_term_memory.recall(user_input)
                        memory_span.span_data.data["recalled_count"] = len(recalled)

                    agent = build_agent(recalled)
                    result = await Runner.run(
                        agent,
                        user_input,
                        context=context,
                        session=session,
                        run_config=RUN_CONFIG,
                        max_turns=12,
                    )
                    response = str(result.final_output)
                    usage = result.context_wrapper.usage
                    with custom_span(
                        "token_usage",
                        data={
                            "estimated_input_tokens": prompt_token_estimate,
                            "api_input_tokens": usage.input_tokens,
                            "api_output_tokens": usage.output_tokens,
                            "api_total_tokens": usage.total_tokens,
                        },
                    ):
                        pass

                    keep, kind, topic = should_remember(user_input)
                    with custom_span(
                        "memory_write_decision",
                        data={
                            "stored": bool(keep and kind),
                            "kind": kind or "none",
                            "topic": topic or "none",
                        },
                    ):
                        if keep and kind:
                            long_term_memory.remember(
                                user_input,
                                kind=kind,
                                topic=topic,
                            )
                            print(
                                f"Long-term memory updated: kind={kind}, topic={topic}"
                            )
                except Exception as error:
                    with custom_span(
                        "request_error",
                        data={"error_type": type(error).__name__},
                    ) as error_span:
                        error_span.set_error(
                            {
                                "message": type(error).__name__,
                                "data": {"stage": "payment_planner_run"},
                            }
                        )
                    raise

            idempotency.complete(session_id, digest, response)
            print("Payment Planner:", response)
            print(
                "Tokens — "
                f"tiktoken estimated input window: {prompt_token_estimate}; "
                f"API input: {usage.input_tokens}; "
                f"API output: {usage.output_tokens}; "
                f"API total: {usage.total_tokens}."
            )
        except Exception as error:
            idempotency.release(session_id, digest)
            print(
                f"Payment Planner: request failed ({type(error).__name__}). "
                "Try again or provide one transaction at a time."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Payment Planner")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--memory-demo", action="store_true")
    parser.add_argument(
        "--langfuse",
        action="store_true",
        help="Export privacy-filtered Agent traces to Langfuse.",
    )
    parser.add_argument("--reset-session", action="store_true")
    parser.add_argument(
        "--session-id",
        default=os.getenv(
            "PAYMENT_PLANNER_SESSION_ID",
            "payment-planner:default:thread-01",
        ),
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.selftest:
        run_tool_selftest()
        return
    if args.memory_demo:
        print("Sliding-window and tiktoken demo:", asyncio.run(memory_demo()))
        with tempfile.TemporaryDirectory() as temporary_directory:
            demo_path = Path(temporary_directory) / "idempotency.db"
            print("Idempotency demo:", idempotency_demo(demo_path))
        return

    try:
        backend = configure_model_backend()
        observability = configure_observability(use_langfuse=args.langfuse)
    except RuntimeError as error:
        print(f"Payment Planner cannot start chat mode: {error}")
        return
    try:
        asyncio.run(
            run_chat(
                args.session_id,
                backend,
                observability,
                reset_session=args.reset_session,
            )
        )
    finally:
        observability.flush()


if __name__ == "__main__":
    main()
