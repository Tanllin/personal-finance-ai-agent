"""Small, explicit configuration shared by startup and Agent Core."""
from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv
from .cache_config import configure_caches

FOLDER = Path(__file__).resolve().parent
load_dotenv(FOLDER / ".env")  # Existing Render environment variables take priority.
os.environ.setdefault("PAYMENT_PLANNER_DATA_DIR", str(FOLDER / "runtime_data"))
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
configure_caches()


@dataclass(frozen=True)
class Settings:
    mode: str
    access_token: str
    data_dir: Path
    user_id: str
    model: str
    timeout_seconds: float
    max_turns: int
    max_output_tokens: int
    rate_limit: int
    compaction: bool
    langfuse: bool

    @classmethod
    def from_env(cls):
        mode = os.getenv("PAYMENT_PLANNER_MODE", "fixture").lower()
        token = os.getenv("PAYMENT_PLANNER_ACCESS_TOKEN", "").strip()
        if mode not in {"fixture", "openai"}:
            raise RuntimeError("PAYMENT_PLANNER_MODE must be fixture or openai.")
        if len(token) < 32:
            raise RuntimeError("Set PAYMENT_PLANNER_ACCESS_TOKEN to a random token of at least 32 characters.")
        if mode == "openai" and not os.getenv("OPENAI_API_KEY", "").strip():
            raise RuntimeError("OpenAI mode requires OPENAI_API_KEY on the server.")
        langfuse = os.getenv("PAYMENT_PLANNER_LANGFUSE", "false").lower() == "true"
        if langfuse and not (os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")):
            raise RuntimeError("Set a reachable LANGFUSE_BASE_URL before enabling Langfuse.")
        config = cls(
            mode, token, Path(os.environ["PAYMENT_PLANNER_DATA_DIR"]),
            os.getenv("PAYMENT_PLANNER_USER_ID", "owner"),
            os.getenv("PAYMENT_PLANNER_MODEL", "gpt-4.1-mini"),
            float(os.getenv("PAYMENT_PLANNER_TIMEOUT_SECONDS", "60")),
            int(os.getenv("PAYMENT_PLANNER_MAX_TURNS", "8")),
            int(os.getenv("PAYMENT_PLANNER_MAX_OUTPUT_TOKENS", "1200")),
            int(os.getenv("PAYMENT_PLANNER_REQUESTS_PER_MINUTE", "20")),
            os.getenv("PAYMENT_PLANNER_COMPACTION", "false").lower() == "true",
            langfuse,
        )
        if not 1 <= config.timeout_seconds <= 120 or not 1 <= config.max_turns <= 12:
            raise RuntimeError("Timeout must be 1–120 seconds and max turns 1–12.")
        if not 1 <= config.max_output_tokens <= 4000 or not 1 <= config.rate_limit <= 60:
            raise RuntimeError("Output tokens must be 1–4000 and requests/minute 1–60.")
        if not config.user_id.strip():
            raise RuntimeError("PAYMENT_PLANNER_USER_ID cannot be empty.")
        return config
