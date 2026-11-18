"""Offline voting/flow tests; run python -m unittest test_semantic_search -v."""
import unittest
from unittest.mock import Mock, patch

from semantic_search import (
    classify_transaction,
    evaluate_neighbors,
    search_similar_transactions,
    seed,
)
from data.transactions import ids


def hits(categories, distances=None):
    distances = distances or [0.1] * len(categories)
    return [{"id": str(i), "document": "demo", "distance": distance,
             "metadata": {"category": category, "verified": True}}
            for i, (category, distance) in enumerate(zip(categories, distances))]


class SemanticTests(unittest.TestCase):
    def test_three_close_votes(self):
        result = evaluate_neighbors(hits(["Dining"] * 3 + ["Travel"] * 2))
        self.assertEqual(result["category"], "Dining")

    def test_weak_majority_abstains(self):
        result = evaluate_neighbors(hits(["Dining"] * 5, [0.8] * 5))
        self.assertTrue(result["needs_review"])

    def test_only_two_close_votes_abstains(self):
        self.assertTrue(evaluate_neighbors(hits(["Dining"] * 5, [0.1, 0.2, 0.7, 0.8, 0.9]))["needs_review"])

    def test_disagreement(self):
        self.assertTrue(evaluate_neighbors(hits(["Dining", "Dining", "Travel", "Travel", "Shopping"]))["needs_review"])

    def test_fewer_than_five(self):
        self.assertTrue(evaluate_neighbors(hits(["Dining"] * 3))["needs_review"])

    def test_unverified_and_duplicate_excluded(self):
        records = hits(["Dining"] * 5)
        records[0]["metadata"]["verified"] = False
        self.assertTrue(evaluate_neighbors(records)["needs_review"])
        records = hits(["Dining"] * 5)
        records[0]["id"] = records[1]["id"]
        self.assertTrue(evaluate_neighbors(records)["needs_review"])

    def test_rule_skips_chroma(self):
        with patch("semantic_search.get_collection", side_effect=AssertionError):
            result = classify_transaction("UBER *TRIP $20.00")
        self.assertEqual(result["category"], "Transportation")
        self.assertEqual(result["transaction"]["amount"], 20)

    def test_invalid_skips_chroma(self):
        with patch("semantic_search.get_collection", side_effect=AssertionError):
            self.assertEqual(classify_transaction("")["status"], "error")
            self.assertEqual(classify_transaction("password demo")["status"], "error")

    def test_flow_filter_and_no_write(self):
        records = hits(["Dining"] * 3 + ["Travel"] * 2)
        collection = Mock()
        collection.query.return_value = {
            key: [[hit[field] for hit in records]] for key, field in
            [("ids", "id"), ("documents", "document"), ("metadatas", "metadata"), ("distances", "distance")]
        }
        result = classify_transaction("NOODLE HOUSE LUNCH $15.00", collection)
        self.assertEqual(result["category"], "Dining")
        self.assertEqual(collection.query.call_args.kwargs["n_results"], 5)
        self.assertEqual(collection.query.call_args.kwargs["where"], {"verified": True})
        collection.add.assert_not_called()
        collection.upsert.assert_not_called()

    def test_failure_abstains(self):
        collection = Mock()
        collection.query.side_effect = RuntimeError("offline")
        self.assertEqual(classify_transaction("UNKNOWN $10.00", collection)["category"], "Other/Uncertain")

    def test_seed_idempotent(self):
        self.assertEqual(len(ids()), len(set(ids())))
        collection = Mock()
        collection.get.return_value = {"ids": ids()}
        seed(collection)
        collection.add.assert_not_called()

    def test_similar_search_skips_chroma_when_local_category_matches(self):
        with patch("semantic_search.get_collection", side_effect=AssertionError):
            result = search_similar_transactions("WHOLE FOODS $42.00")

        self.assertFalse(result["search_performed"])
        self.assertEqual(result["suggested_category"], "Groceries")

    def test_similar_search_returns_verified_neighbors_for_other(self):
        records = hits(["Dining"] * 3 + ["Travel"] * 2)
        collection = Mock()
        collection.query.return_value = {
            key: [[hit[field] for hit in records]] for key, field in
            [("ids", "id"), ("documents", "document"),
             ("metadatas", "metadata"), ("distances", "distance")]
        }

        result = search_similar_transactions(
            "NOODLE HOUSE SPECIAL $15.00",
            collection,
        )

        self.assertTrue(result["search_performed"])
        self.assertEqual(result["suggested_category"], "Dining")
        self.assertEqual(len(result["neighbors"]), 5)
        collection.add.assert_not_called()
        collection.upsert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
