"""One reusable message workflow, independent of HTTP and the webpage.

Single owner, one process. Successful request IDs are cached; identical text
with a NEW ID is intentionally a new turn. Failed runs can be retried, but are
not guaranteed exactly-once across a crash (there are no payment side effects).
"""
from __future__ import annotations

import asyncio
from contextlib import closing, contextmanager
from dataclasses import asdict
import hashlib
import json
import sqlite3
import time

from agents import RunConfig, Runner, SQLiteSession, custom_span, trace

from .settings import Settings
from PaymentPlanner.agent import build_agent
from PaymentPlanner.long_term_memory import (
    LongTermPreferenceMemory, contains_sensitive_input, should_remember,
)
from PaymentPlanner.memory import (
    KEEP_ITEMS, RUN_CONFIG, count_items_tokens, create_compaction_session,
    load_planner_context,
)
from PaymentPlanner.observability import trace_group_id
from PaymentPlanner.tools import parse_transaction, categorize_transaction


class CoreError(Exception):
    def __init__(self, status: int, message: str):
        self.status, self.message = status, message
        super().__init__(message)


class AgentCore:
    def __init__(self, settings: Settings):
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.lock = asyncio.Lock()
        self.memory = None  # Chroma initializes only in live mode, at runtime.
        self.db_path = settings.data_dir / "web_requests.db"
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS requests (session TEXT, request_id TEXT, input_hash TEXT, response TEXT, created REAL, PRIMARY KEY(session, request_id))")

    @contextmanager
    def _db(self):
        with closing(sqlite3.connect(self.db_path, timeout=10)) as db:
            with db:
                yield db

    def session_id(self, thread_id: str) -> str:
        return "web-" + hashlib.sha256(f"{self.settings.user_id}:{thread_id}".encode()).hexdigest()

    async def chat(self, message: str, thread_id: str, request_id: str) -> dict:
        message = message.strip()
        if not message or len(message) > 6000:
            raise CoreError(400, "Provide a message of 1–6000 characters.")
        # Before traces, caches, model requests, or either memory layer.
        if contains_sensitive_input(message):
            raise CoreError(400, "Sensitive banking credentials are not accepted. Remove credentials and full account/card numbers.")
        try:
            async with asyncio.timeout(self.settings.timeout_seconds):
                async with self.lock:
                    return await self._chat(message, thread_id, request_id)
        except TimeoutError:
            with trace("payment_planner_timeout"):
                with custom_span("request_error") as span:
                    span.set_error({"message": "TimeoutError", "data": {}})
            raise CoreError(504, "The planner timed out. Please try again.") from None

    async def _chat(self, message: str, thread_id: str, request_id: str) -> dict:
        session_id = self.session_id(thread_id)
        digest = hashlib.sha256(message.encode()).hexdigest()
        with self._db() as db:
            db.execute("DELETE FROM requests WHERE created < ?", (time.time() - 86400,))
            cached = db.execute("SELECT input_hash,response FROM requests WHERE session=? AND request_id=?", (session_id, request_id)).fetchone()
        if cached:
            if cached[0] != digest:
                raise CoreError(409, "This request ID was already used for a different message.")
            return {**json.loads(cached[1]), "cached": True}
        session = SQLiteSession(session_id, str(self.settings.data_dir / "sessions.db"))
        try:
            return await self._run(message, thread_id, request_id, session_id, digest, session)
        finally:
            session.close()

    async def _run(self, message, thread_id, request_id, session_id, digest, session):
        started = time.monotonic()
        context = load_planner_context(self.settings.data_dir / "planner_profile.json")
        # An internal runtime-only path; not part of saved/profile JSON or prompts.
        context._profile_path = self.settings.data_dir / "planner_profile.json"
        warnings = []
        estimated = None
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        tools_used = []
        remembered = False
        with trace("payment_planner_web", group_id=trace_group_id(session_id)):
            try:
                history = await session.get_items(limit=100)
                try:
                    estimated = count_items_tokens(history[-KEEP_ITEMS:] + [{"role": "user", "content": message}])
                except Exception:
                    warnings.append("Local tiktoken estimate unavailable; API usage is reported separately.")
                if self.settings.mode == "fixture":
                    parsed = parse_transaction(message)
                    categorized = ({**parsed, **categorize_transaction(parsed["description"], parsed.get("merchant"))}
                                   if parsed.get("status") == "ok" else parsed)
                    answer = "Offline fixture: deterministic parsing/categorization only; no model, semantic retrieval, or long-term memory.\n" + json.dumps(categorized, indent=2)
                    tools_used = ["parse_transaction", "categorize_transaction"]
                    await session.add_items([{"role": "user", "content": message}, {"role": "assistant", "content": answer}])
                else:
                    recalled = []
                    with custom_span("memory_retrieval") as span:
                        try:
                            if self.memory is None:
                                self.memory = LongTermPreferenceMemory(self.settings.user_id)
                            recalled = self.memory.recall(message)
                        except Exception:
                            warnings.append("Long-term recall unavailable; structured profile and SQLite history still apply.")
                        span.span_data.data = {"recalled_count": len(recalled)}
                    agent = build_agent(recalled, model=self.settings.model, max_output_tokens=self.settings.max_output_tokens)
                    run_session = session
                    if self.settings.compaction:
                        run_session = create_compaction_session(session_id, session, self.settings.model)
                    config = RunConfig(
                        session_input_callback=RUN_CONFIG.session_input_callback,
                        session_settings=RUN_CONFIG.session_settings,
                        trace_include_sensitive_data=False,
                    )
                    result = await Runner.run(agent, message, context=context, session=run_session,
                                              run_config=config, max_turns=self.settings.max_turns)
                    answer = str(result.final_output)
                    api_usage = result.context_wrapper.usage
                    usage = {key: getattr(api_usage, key) for key in usage}
                    raw_items = [item.raw_item if isinstance(item.raw_item, dict)
                                 else item.raw_item.model_dump() for item in result.new_items]
                    tools_used = list(dict.fromkeys(
                        item.get("name") for item in raw_items
                        if item.get("type") == "function_call" and item.get("name")
                    ))
                    keep, kind, topic = should_remember(message)
                    with custom_span("memory_write_decision") as span:
                        if keep and kind and self.memory is not None:
                            try:
                                self.memory.remember(message, kind=kind, topic=topic)
                                remembered = True
                            except Exception:
                                warnings.append("Long-term preference could not be saved.")
                        span.span_data.data = {"stored": remembered}
                with custom_span("token_usage", data=usage):
                    pass
            except Exception as error:
                with custom_span("request_error") as span:
                    span.set_error({"message": type(error).__name__, "data": {}})
                # Never return provider exceptions or log message/credentials.
                raise CoreError(503, "The planner could not complete this request. Check server configuration and retry.") from None
        response = {
            "answer": answer, "thread_id": thread_id, "request_id": request_id,
            "mode": self.settings.mode, "cached": False, "tools_used": tools_used,
            "profile": asdict(context), "remembered": remembered,
            "usage": usage, "estimated_history_and_message_tokens": estimated,
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            "warnings": warnings,
        }
        with self._db() as db:
            db.execute("INSERT INTO requests VALUES (?,?,?,?,?)", (session_id, request_id, digest, json.dumps(response), time.time()))
        return response

    async def reset(self, thread_id: str):
        async with self.lock:
            session_id = self.session_id(thread_id)
            session = SQLiteSession(session_id, str(self.settings.data_dir / "sessions.db"))
            try:
                await session.clear_session()
            finally:
                session.close()
            with self._db() as db:
                db.execute("DELETE FROM requests WHERE session=?", (session_id,))
        return {"status": "ok", "thread_id": thread_id,
                "message": "Conversation and its retry cache cleared. Profile and long-term preferences are unchanged."}
