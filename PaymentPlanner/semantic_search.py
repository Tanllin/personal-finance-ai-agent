"""Rule-first classification, then Chroma retrieval + a distance-gated vote.

Like Lesson 08 demos 01-05: embed -> seed -> metadata filter -> distance gate
-> evidence-backed answer. No chat model, banking connection, or payment tools.

Run: python semantic_search.py 'CITY RAIL DAY TICKET $8.00'
Default: demonstrate one rule, one semantic query, and one unrelated query.
Chroma's local all-MiniLM-L6-v2 embedding model downloads on first semantic use.
"""

from __future__ import annotations

import argparse
from collections import Counter
import math
import os
from pathlib import Path
import re

if __package__:
    from .data.transactions import ids, texts, metas
    from .tools import categorize_transaction, parse_transaction
else:
    from data.transactions import ids, texts, metas
    from tools import categorize_transaction, parse_transaction

TOP_K = 5
# Among the top 5 results, at least 3 need to agree on the same category
MIN_AGREEMENT = 3
# Only those whose distance is smaller than this threshold can vote
MAX_DISTANCE = 0.35
DB_PATH = Path(os.getenv("PAYMENT_PLANNER_DATA_DIR", str(Path(__file__).resolve().parent / "data"))) / "transaction_chroma"
COLLECTION_NAME = "verified-transactions-minilm-cosine-v1"

# Narrow descriptor rules only. The older categorizer labels every substring
# match 0.95, which is not sufficient evidence of high confidence (e.g. rent
# inside a different word). Broad merchant names intentionally fall through.
RULES = [
    (r"^UBER\s*\*\s*TRIP\b", "Transportation"),
    (r"^UBER\s*\*?\s*EATS\b", "Dining"),
    (r"^LYFT\s*\*\s*RIDE\b", "Transportation"),
    (r"^NETFLIX(?:\s|\*|$)", "Entertainment"),
    (r"^PAYROLL\s+DIRECT\s+DEPOSIT\b", "Income"),
]

# Provide ChromaDB collection where the verified transactions are stored
def get_collection():
    """Open real persistent Chroma with a fixed local semantic embedding model."""
    import chromadb
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=DefaultEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )

# Take the verified examples from transaction.py and add any that aren't alreadey in ChromaDB
def seed(collection) -> None:
    """Add missing fixed example IDs only; never save a query or its prediction."""
    existing = set(collection.get(ids=ids(), include=[])["ids"])
    rows = [(i, d, m) for i, d, m in zip(ids(), texts(), metas()) if i not in existing]
    if rows:
        collection.add(
            ids=[row[0] for row in rows],
            documents=[row[1] for row in rows],
            metadatas=[row[2] for row in rows],
        )

# Taking an unknown transaction description and ask ChromaDB for the 5 most sementically similar verified transactions
def retrieve(description: str, collection) -> list[dict]:
    """Ask for the top five *verified* neighbors and retain evidence for review."""
    result = collection.query(
        query_texts=[description], n_results=TOP_K,
        where={"verified": True},
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"id": identifier, "document": doc, "metadata": meta, "distance": distance}
        for identifier, doc, meta, distance in zip(
            result["ids"][0], result["documents"][0],
            result["metadatas"][0], result["distances"][0],
        )
    ]


def search_similar_transactions(user_input: str, collection=None) -> dict:
    """Search verified Chroma examples only when local categorization is Other.

    The query and its prediction are never written to ChromaDB. The result includes
    the five nearest verified examples plus the same agreement/distance decision
    used by semantic classification.
    """

    parsed = parse_transaction(user_input)
    if parsed["status"] != "ok" or not parsed.get("description"):
        return {
            "status": "error",
            "source": "parser",
            "search_performed": False,
            "transaction": parsed,
            "neighbors": [],
            "suggested_category": "Other/Uncertain",
            "needs_review": True,
            "reason": "A valid transaction description is required.",
        }

    local = categorize_transaction(
        parsed["description"],
        parsed.get("merchant"),
    )
    if local["category"] != "Other":
        return {
            "status": "ok",
            "source": "local_rule",
            "search_performed": False,
            "transaction": parsed,
            "neighbors": [],
            "suggested_category": local["category"],
            "needs_review": False,
            "reason": "A deterministic local category matched; semantic fallback was not needed.",
        }

    try:
        if collection is None:
            collection = get_collection()
            seed(collection)
        neighbors = retrieve(parsed["description"], collection)
        decision = evaluate_neighbors(neighbors)
        return {
            "status": "ok",
            "source": "semantic",
            "search_performed": True,
            "transaction": parsed,
            "neighbors": neighbors,
            "suggested_category": decision["category"],
            "needs_review": decision["needs_review"],
            "close_category_votes": decision["close_category_votes"],
            "max_distance": decision["max_distance"],
            "reason": decision["reason"],
        }
    except Exception as error:
        return {
            "status": "error",
            "source": "semantic",
            "search_performed": True,
            "transaction": parsed,
            "neighbors": [],
            "suggested_category": "Other/Uncertain",
            "needs_review": True,
            "reason": (
                f"Semantic retrieval unavailable ({type(error).__name__}). "
                "Check dependencies/model cache."
            ),
        }

# This function are deciding are these 5 results strong enough and consistent enough for one to trust a category

def evaluate_neighbors(neighbors: list[dict], max_distance: float = MAX_DISTANCE) -> dict:
    """
    This function check whether the top give ChromaDB results provide enough evidence to assign a transaction category

    Specifically, it will first removes unverified or uncertain exmaples, along with keeping only 
    neighbors within the maximum distance threshold. 
    The remaining neighbors vote based on their category, and at least 3 of the 5 nighbors must 
    agree on the same category for the classification to be accepted
    
    Otherwise, the transaction is returned as Other/Uncertain
    """
    if not math.isfinite(max_distance) or not 0 <= max_distance <= 2:
        raise ValueError("Cosine-distance threshold must be finite and between 0 and 2.")
    eligible = [
        hit for hit in neighbors[:TOP_K]
        if hit.get("metadata", {}).get("verified") is True
        and hit.get("metadata", {}).get("category") not in {None, "", "Other", "Other/Uncertain"}
    ]
    close = [
        hit for hit in eligible
        if isinstance(hit.get("distance"), (int, float))
        and math.isfinite(hit["distance"])
        and hit["distance"] <= max_distance
    ]
    votes = Counter(hit["metadata"]["category"] for hit in close)
    winner, count = votes.most_common(1)[0] if votes else (None, 0)
    complete = len(eligible) == TOP_K and len({hit["id"] for hit in eligible}) == TOP_K
    accepted = complete and count >= MIN_AGREEMENT
    return {
        "category": winner if accepted else "Other/Uncertain",
        "needs_review": not accepted,
        "close_category_votes": dict(votes),
        "max_distance": max_distance,
        "reason": (
            "At least three of five verified examples agree within the distance threshold."
            if accepted else "Insufficient verified neighbors, category agreement, or similarity."
        ),
    }


def classify_transaction(user_input: str, collection=None,
                         max_distance: float = MAX_DISTANCE) -> dict:
    """
    This function will first obtain parse_transaction() from the tools box, returning with structrued
    transaction information

    It will then check the rule, if rule matches, then it wil lnot called ChromaDB

    If the rule does not match, it will use ChromaDB

    Then decide whether the top 5 are trustworthy
    """
    parsed = parse_transaction(user_input)
    base = {"transaction": parsed, "neighbors": []}
    if parsed["status"] != "ok" or not parsed.get("description"):
        return {**base, "status": "error", "source": "parser",
                "category": "Other/Uncertain", "needs_review": True,
                "reason": "A valid transaction description is required."}

    description = parsed["description"]
    for pattern, category in RULES:
        if re.search(pattern, description, re.IGNORECASE):
            return {**base, "status": "ok", "source": "rule",
                    "category": category, "needs_review": False, "matched_rule": pattern}

    try:
        if collection is None:
            collection = get_collection()
            seed(collection)
        neighbors = retrieve(description, collection)
        decision = evaluate_neighbors(neighbors, max_distance)
        return {**base, **decision, "neighbors": neighbors,
                "status": "ok", "source": "semantic"}
    except Exception as error:
        # Do not pretend keyword vectors or database errors are semantic evidence.
        return {**base, "status": "error", "source": "semantic",
                "category": "Other/Uncertain", "needs_review": True,
                "reason": f"Semantic retrieval unavailable ({type(error).__name__}). Check dependencies/model cache."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transaction", nargs="?")
    parser.add_argument("--max-distance", type=float, default=MAX_DISTANCE)
    args = parser.parse_args()
    queries = [args.transaction] if args.transaction else [
        "UBER *TRIP $20.00", "CITY RAIL DAY TICKET $8.00",
        "QUANTUM PARTICLE RESEARCH MATERIAL $50.00",
    ]
    for query in queries:
        result = classify_transaction(query, max_distance=args.max_distance)
        print(f"\nCategory: {result['category']} | source: {result['source']} | review: {result['needs_review']}")
        for hit in result["neighbors"]:
            print(f"  {hit['id']} distance={hit['distance']:.4f} [{hit['metadata']['category']}] {hit['document']}")
        print(result.get("reason", "Matched a precise known descriptor rule."))


if __name__ == "__main__":
    main()
