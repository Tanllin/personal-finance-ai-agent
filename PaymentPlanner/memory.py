"""Session history, context-window policy, and structured planner memory."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import tiktoken
from agents import (
    RunConfig,
    RunContextWrapper,
    SessionSettings,
    SQLiteSession,
    function_tool,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("PAYMENT_PLANNER_DATA_DIR", str(PROJECT_DIR / "data")))
SESSION_DB_PATH = DATA_DIR / "sessions.db"
PROFILE_PATH = DATA_DIR / "planner_profile.json"
IDEMPOTENCY_DB_PATH = DATA_DIR / "idempotency.db"

# For this, we are keeping about 100 turns of message from one sessions
# and letting the context window to be 6
MAX_STORED_ITEMS = 100
KEEP_ITEMS = 6
TOKEN_ENCODING = "cl100k_base"

# The below is an example of structrued application of state containing infomration that
# might be important for the agent to remember, and can also work even when conversation 
# message disappear from the context window 
@dataclass
class PaymentPlannerContext:
    """Durable non-sensitive facts that should survive context trimming."""

    default_currency: str = "USD"
    monthly_budget: float | None = None
    savings_goal: float | None = None


def ensure_data_directory() -> None:
    """Create the local data directory when the application starts."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

# This is using the state obtained from PaymentPlannerContext and convert that to readable message
# to the model
def planner_profile_summary(context: PaymentPlannerContext) -> str:
    """Format structured memory for injection into the Agent instructions."""

    budget = (
        f"{context.monthly_budget:.2f} {context.default_currency}"
        if context.monthly_budget is not None
        else "not set"
    )
    savings_goal = (
        f"{context.savings_goal:.2f} {context.default_currency}"
        if context.savings_goal is not None
        else "not set"
    )
    return (
        f"Default currency: {context.default_currency}\n"
        f"Monthly budget: {budget}\n"
        f"Savings goal: {savings_goal}\n"
        "Sensitive banking credentials are never stored in this profile."
    )


def load_planner_context(path: Path = PROFILE_PATH) -> PaymentPlannerContext:
    """Load persisted profile facts, falling back safely to defaults."""

    if not path.exists():
        return PaymentPlannerContext()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return PaymentPlannerContext(
            default_currency=str(payload.get("default_currency", "USD")).upper(),
            monthly_budget=(
                float(payload["monthly_budget"])
                if payload.get("monthly_budget") is not None
                else None
            ),
            savings_goal=(
                float(payload["savings_goal"])
                if payload.get("savings_goal") is not None
                else None
            ),
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return PaymentPlannerContext()


def save_planner_context(
    context: PaymentPlannerContext,
    path: Path = PROFILE_PATH,
) -> None:
    """Persist only the supported non-sensitive planner facts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(context), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def update_planner_preference(
    context: PaymentPlannerContext,
    field: Literal["default_currency", "monthly_budget", "savings_goal"],
    value: str,
    path: Path = PROFILE_PATH,
) -> dict:
    """Validate and persist one supported planner preference."""

    cleaned_value = (value or "").strip()
    if field == "default_currency":
        currency = cleaned_value.upper()
        if not re_full_currency_code(currency):
            return {
                "status": "error",
                "message": "Currency must be a three-letter code such as USD or EUR.",
            }
        context.default_currency = currency
    elif field in {"monthly_budget", "savings_goal"}:
        try:
            amount = float(cleaned_value.replace(",", ""))
        except ValueError:
            return {
                "status": "error",
                "message": f"{field} must be a positive number.",
            }
        if not math.isfinite(amount) or amount <= 0:
            return {
                "status": "error",
                "message": f"{field} must be greater than zero.",
            }
        setattr(context, field, amount)
    else:
        return {"status": "error", "message": f"Unsupported field: {field}"}

    save_planner_context(context, path)
    return {
        "status": "ok",
        "field": field,
        "value": getattr(context, field),
        "message": "Planner preference saved.",
    }


def re_full_currency_code(value: str) -> bool:
    """Return whether a value resembles an ISO-style three-letter code."""

    return len(value) == 3 and value.isalpha() and value.isascii()


@function_tool
def save_planner_preference(
    wrapper: RunContextWrapper[PaymentPlannerContext],
    field: Literal["default_currency", "monthly_budget", "savings_goal"],
    value: str,
) -> dict:
    """Save an explicitly stated non-sensitive Payment Planner preference.

    Use this only when the user explicitly gives or changes a default currency or
    monthly budget, or savings goal. Never use it for transaction details,
    credentials, account numbers, card information, passwords, PINs, CVVs, or
    authentication data.

    Args:
        field: ``default_currency``, ``monthly_budget``, or ``savings_goal``.
        value: Currency code or positive amount supplied by the user.
    """

    return update_planner_preference(
        wrapper.context, field, value,
        path=getattr(wrapper.context, "_profile_path", PROFILE_PATH),
    )

# This is to define the number of turns, which is 6
def keep_recent_history(
    history: list[Any],
    new_input: list[Any],
) -> list[Any]:
    """Send recent history to the model while retaining more in SQLite."""

    return history[-KEEP_ITEMS:] + new_input


def _tokenizable_text(value: Any) -> str:
    """Turn an SDK item or plain value into stable text for token estimation."""

    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)

# This function takes a list of conversation messages and adds up the token count of every message
def count_tokens(value: Any) -> int:
    """Count text tokens locally with tiktoken's cl100k_base encoding."""

    encoding = tiktoken.get_encoding(TOKEN_ENCODING)
    return len(encoding.encode(_tokenizable_text(value)))

# this is just counting the total number of tokens
def count_items_tokens(items: list[Any]) -> int:
    """Estimate total tokens for a list of conversation items."""

    return sum(count_tokens(item) for item in items)


RUN_CONFIG = RunConfig(
    session_input_callback=keep_recent_history,
    session_settings=SessionSettings(limit=MAX_STORED_ITEMS),
)


def create_session(session_id: str) -> SQLiteSession:
    """Create a stable SQLite-backed session for one conversation thread."""

    ensure_data_directory()
    return SQLiteSession(session_id, str(SESSION_DB_PATH))


# The below create a compaction-enabled session, but it does not itself force a compaction
# Notice the SDK rule is to automatically compact when there are at least 10 compaction candidate
# items
# TODO: Decide to let it automatically decide when to compact(like human interference)
def create_compaction_session(
    session_id: str,
    underlying_session: SQLiteSession,
    model: str = "gpt-4.1",
):
    """Wrap SQLite history with OpenAI Responses compaction.

    Compaction is intentionally separate from structured profile memory because
    summaries are lossy. It is enabled only for OpenAI Responses-compatible chat.
    """

    from agents.memory import OpenAIResponsesCompactionSession

    return OpenAIResponsesCompactionSession(
        session_id=session_id,
        underlying_session=underlying_session,
        model=model,
    )


class IdempotencyStore:
    """Prevent identical normalized input from being processed twice per session."""

    # The below function set up the idempotency system, which creates the database and makes sure 
    # the tbale is tacking requests
    def __init__(self, path: Path = IDEMPOTENCY_DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as database:
            database.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_inputs (
                    session_id TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    normalized_input TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response TEXT,
                    PRIMARY KEY (session_id, input_hash)
                )
                """
            )
    # Helper function to help connect
    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    # Small difference are ignored
    @staticmethod
    def normalize(user_input: str) -> str:
        return " ".join((user_input or "").split()).casefold()

    @classmethod
    def input_hash(cls, user_input: str) -> str:
        normalized = cls.normalize(user_input)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    # This is to check whether the requests should be processed, and can return a total of three
    def begin(self, session_id: str, user_input: str) -> tuple[str, str, str | None]:
        """Reserve an input or return its prior completed response."""

        normalized = self.normalize(user_input)
        digest = self.input_hash(user_input)
        try:
            with self._connect() as database:
                database.execute(
                    "INSERT INTO processed_inputs VALUES (?, ?, ?, 'pending', NULL)",
                    (session_id, digest, normalized),
                )
            return "new", digest, None
        except sqlite3.IntegrityError:
            with self._connect() as database:
                row = database.execute(
                    """
                    SELECT status, response FROM processed_inputs
                    WHERE session_id = ? AND input_hash = ?
                    """,
                    (session_id, digest),
                ).fetchone()
            if row and row[0] == "complete":
                return "duplicate", digest, row[1]
            return "pending", digest, None
    
    # Marks a request as successfully finished and saves its response
    def complete(self, session_id: str, digest: str, response: str) -> None:
        with self._connect() as database:
            database.execute(
                """
                UPDATE processed_inputs SET status = 'complete', response = ?
                WHERE session_id = ? AND input_hash = ?
                """,
                (response, session_id, digest),
            )

    def release(self, session_id: str, digest: str) -> None:
        """Release a failed pending request so the user can retry it."""

        with self._connect() as database:
            database.execute(
                """
                DELETE FROM processed_inputs
                WHERE session_id = ? AND input_hash = ? AND status = 'pending'
                """,
                (session_id, digest),
            )

    def clear_session(self, session_id: str) -> None:
        with self._connect() as database:
            database.execute(
                "DELETE FROM processed_inputs WHERE session_id = ?",
                (session_id,),
            )


async def memory_demo() -> dict:
    """Offline proof that storage and model context have separate limits."""

    session = create_session("payment-planner:memory-demo")
    await session.clear_session()
    for turn in range(1, 9):
        await session.add_items(
            [
                {"role": "user", "content": f"Turn {turn}: transaction question"},
                {"role": "assistant", "content": f"Turn {turn}: recorded"},
            ]
        )

    stored = await session.get_items()
    new_input = [{"role": "user", "content": "Use my latest information."}]
    model_input = keep_recent_history(stored, new_input)

    assert len(stored) == 16
    assert len(model_input) == KEEP_ITEMS + 1
    return {
        "stored_items": len(stored),
        "model_input_items": len(model_input),
        "keep_items": KEEP_ITEMS,
        "stored_tokens_tiktoken": count_items_tokens(stored),
        "model_input_tokens_tiktoken": count_items_tokens(model_input),
    }


def idempotency_demo(path: Path) -> dict:
    """Offline proof that repeated normalized input is processed once."""

    store = IdempotencyStore(path)
    session_id = "payment-planner:idempotency-demo"
    first_status, digest, _ = store.begin(session_id, "UBER *TRIP $20.00")
    store.complete(session_id, digest, "Transportation")
    second_status, _, cached = store.begin(session_id, "  uber *trip $20.00  ")
    assert first_status == "new"
    assert second_status == "duplicate"
    assert cached == "Transportation"
    return {
        "first_input": first_status,
        "same_normalized_input": second_status,
        "cached_response": cached,
    }


if __name__ == "__main__":
    print(asyncio.run(memory_demo()))
