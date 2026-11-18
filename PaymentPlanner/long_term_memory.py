"""P3 long-term preference memory integrated from Lesson 09's 05 example.

Only durable, non-sensitive preferences are promoted. Raw transactions remain
working data. A stable topic is used as the Chroma ID, so a correction replaces
the earlier value for that user and topic.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


DATA_PATH = Path(os.getenv("PAYMENT_PLANNER_DATA_DIR", str(Path(__file__).resolve().parent / "data"))) / "long_term_memory"
COLLECTION_NAME = "payment-planner-preferences-minilm-v1"
RECALL_TOP_K = 3
RECALL_MAX_DISTANCE: float | None = None
MAX_RECALLED_CHARACTERS = 900

SENSITIVE_PATTERN = re.compile(
    r"\b(?:password|passcode|pin|routing number|account number|cvv|security code|"
    r"authentication token|security answer)\b",
    re.IGNORECASE,
)
POSSIBLE_FULL_CARD_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
TRANSACTION_AMOUNT_PATTERN = re.compile(
    r"(?:[$€£]\s*\d|\b\d[\d,]*\.\d{2}\s*(?:USD|EUR|GBP|CAD|AUD)?\b)",
    re.IGNORECASE,
)


def contains_sensitive_input(text: str) -> bool:
    """Reject likely credentials before model calls or either memory layer."""

    return bool(
        SENSITIVE_PATTERN.search(text or "")
        or POSSIBLE_FULL_CARD_PATTERN.search(text or "")
    )


def should_remember(text: str) -> tuple[bool, str | None, str | None]:
    """Return (keep, kind, topic) for explicit durable preferences only."""

    clean = " ".join((text or "").split())
    lower = clean.casefold()
    if not clean or contains_sensitive_input(clean):
        return False, None, None

    correction = re.search(
        r"(?:categorize|classify|treat)\s+(.{1,50}?)\s+as\s+"
        r"(transportation|groceries|dining|utilities|shopping|entertainment|"
        r"health|travel|housing|fees|income|transfers)",
        clean,
        re.IGNORECASE,
    )
    if correction:
        merchant_key = re.sub(
            r"[^a-z0-9]+", "_", correction.group(1).casefold()
        ).strip("_")
        return True, "category_preference", f"merchant_category:{merchant_key}"

    routes = [
        (
            "preference",
            "default_currency",
            ("default currency", "currency is", "use usd", "use eur", "用美元", "用欧元"),
        ),
        (
            "constraint",
            "monthly_budget",
            ("monthly budget", "budget is", "每月预算", "月预算"),
        ),
        (
            "goal",
            "savings_goal",
            ("savings goal", "save each month", "want to save", "储蓄目标", "每月存"),
        ),
        (
            "preference",
            "report_style",
            ("show me tables", "use tables", "brief summary", "用表格", "简短总结"),
        ),
    ]
    for kind, topic, hints in routes:
        if any(hint in lower for hint in hints):
            return True, kind, topic

    # Transaction rows are working data, not permanent user-profile facts.
    if TRANSACTION_AMOUNT_PATTERN.search(clean):
        return False, None, None
    return False, None, None


class LongTermPreferenceMemory:
    """User-scoped Chroma memory with topic-based upsert and deletion."""

    def __init__(self, user_id: str, collection: str = COLLECTION_NAME):
        self.user_id = user_id
        self._client = chromadb.PersistentClient(path=str(DATA_PATH))
        self._collection = self._client.get_or_create_collection(
            name=collection,
            embedding_function=DefaultEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )

    def _memory_id(self, text: str, topic: str | None) -> str:
        key = f"{self.user_id}:topic:{topic}" if topic else f"{self.user_id}:{text}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]

    def remember(self, text: str, kind: str, topic: str | None) -> str:
        """Upsert a routed preference; never call this for unapproved input."""

        memory_id = self._memory_id(text, topic)
        metadata = {
            "user_id": self.user_id,
            "kind": kind,
            "topic": topic or "unscoped",
        }
        self._collection.upsert(
            ids=[memory_id],
            documents=[text],
            metadatas=[metadata],
        )
        return memory_id

    def recall(
        self,
        query: str,
        top_k: int = RECALL_TOP_K,
        max_distance: float | None = RECALL_MAX_DISTANCE,
    ) -> list[dict]:
        """Recall a small user-scoped set and optionally reject distant hits."""

        available = self.count()
        if available == 0:
            return []
        result = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, available),
            where={"user_id": self.user_id},
        )
        recalled = []
        for text, metadata, distance in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            if max_distance is not None and distance > max_distance:
                continue
            recalled.append(
                {
                    "text": text,
                    "kind": metadata.get("kind", "fact"),
                    "topic": metadata.get("topic"),
                    "distance": distance,
                }
            )
        return recalled

    def count(self) -> int:
        return len(self._collection.get(where={"user_id": self.user_id})["ids"])

    def peek_topic(self, topic: str) -> str | None:
        result = self._collection.get(
            where={"$and": [{"user_id": self.user_id}, {"topic": topic}]}
        )
        documents = result.get("documents", [])
        return documents[0] if documents else None

    def count_topic(self, topic: str) -> int:
        result = self._collection.get(
            where={"$and": [{"user_id": self.user_id}, {"topic": topic}]}
        )
        return len(result.get("ids", []))

    def forget(self, topic: str) -> int:
        result = self._collection.get(
            where={"$and": [{"user_id": self.user_id}, {"topic": topic}]}
        )
        memory_ids = result.get("ids", [])
        if memory_ids:
            self._collection.delete(ids=memory_ids)
        return len(memory_ids)


def build_recalled_memory_block(recalled: list[dict]) -> str:
    """Deduplicate and bound recalled memory before prompt fusion."""

    lines = []
    seen = set()
    used_characters = 0
    for hit in recalled:
        key = (hit.get("topic"), hit["text"])
        if key in seen:
            continue
        line = f"- ({hit['kind']}, topic={hit.get('topic')}) {hit['text']}"
        if used_characters + len(line) > MAX_RECALLED_CHARACTERS:
            break
        seen.add(key)
        lines.append(line)
        used_characters += len(line)
    return "\n".join(lines)
