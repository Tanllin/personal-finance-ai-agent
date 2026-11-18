"""Offline tests for Payment Planner analytics tools.

Run: python -m unittest -v test_analytics_tools.py
"""

import unittest

from tools import (
    aggregate_monthly_spending,
    calculate_net_cash_flow,
    compare_monthly_spending,
    detect_recurring_transactions,
    get_transactions,
    summarize_accounts,
)


def transaction(
    transaction_date: str,
    merchant: str,
    amount: float,
    category: str,
    currency: str = "USD",
    account: str | None = None,
) -> dict:
    record = {
        "transaction_date": transaction_date,
        "merchant": merchant,
        "amount": amount,
        "currency": currency,
        "category": category,
    }
    if account is not None:
        record["account"] = account
    return record


class AnalyticsToolTests(unittest.TestCase):
    def test_aggregate_groups_spending_and_excludes_income_and_transfers(self):
        rows = [
            transaction("2026-08-01", "Cafe", 15, "Dining"),
            transaction("2026-08-02", "Cafe Refund", -5, "Dining"),
            transaction("2026-08-03", "Market", 50, "Groceries"),
            transaction("2026-08-04", "Employer", 3000, "Income"),
            transaction("2026-08-05", "Savings", 500, "Transfers"),
            transaction("2026-07-31", "Old Cafe", 90, "Dining"),
            transaction("2026-08-06", "Paris Cafe", 12, "Dining", "EUR"),
        ]

        result = aggregate_monthly_spending(rows, "2026-08")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["selected_transaction_count"], 6)
        self.assertEqual(result["excluded_income_or_transfer_count"], 2)
        self.assertEqual(result["totals_by_currency"]["USD"]["Dining"], 10)
        self.assertEqual(result["spending_total_by_currency"]["USD"], 60)
        self.assertEqual(result["spending_total_by_currency"]["EUR"], 12)

    def test_net_cash_flow_treats_refund_as_reduced_expense(self):
        rows = [
            transaction("2026-08-01", "Employer", 3000, "Income"),
            transaction("2026-08-02", "Landlord", 1200, "Housing"),
            transaction("2026-08-03", "Store Refund", -50, "Shopping"),
            transaction("2026-08-04", "Savings", 500, "Transfers"),
        ]

        result = calculate_net_cash_flow(rows, "2026-08")["by_currency"]["USD"]

        self.assertEqual(result["income"], 3000)
        self.assertEqual(result["expenses"], 1150)
        self.assertEqual(result["net_cash_flow"], 1850)
        self.assertEqual(result["excluded_transfer_count"], 1)

    def test_compare_months_handles_new_and_stopped_categories(self):
        rows = [
            transaction("2026-07-01", "Cafe", 100, "Dining"),
            transaction("2026-07-02", "Market", 50, "Groceries"),
            transaction("2026-08-01", "Cafe", 120, "Dining"),
            transaction("2026-08-02", "Metro", 20, "Transportation"),
        ]

        comparison = compare_monthly_spending(
            rows, "2026-08", "2026-07"
        )["comparison_by_currency"]["USD"]

        self.assertEqual(comparison["Dining"]["percent_change"], 20)
        self.assertEqual(comparison["Dining"]["trend"], "increased")
        self.assertEqual(comparison["Groceries"]["trend"], "stopped")
        self.assertEqual(comparison["Groceries"]["percent_change"], -100)
        self.assertEqual(comparison["Transportation"]["trend"], "new")
        self.assertIsNone(comparison["Transportation"]["percent_change"])

    def test_recurring_finds_consistent_merchant_and_reports_price_change(self):
        rows = [
            transaction("2026-01-15", "Netflix", 10, "Entertainment"),
            transaction("2026-02-15", "NETFLIX", 10, "Entertainment"),
            transaction("2026-03-15", "Netflix", 12, "Entertainment"),
            transaction("2026-01-02", "Uber", 20, "Transportation"),
            transaction("2026-02-20", "Uber", 18, "Transportation"),
            transaction("2026-01-01", "Savings", 100, "Transfers"),
            transaction("2026-02-01", "Savings", 100, "Transfers"),
        ]

        result = detect_recurring_transactions(rows)

        self.assertEqual(result["recurring_count"], 1)
        pattern = result["patterns"][0]
        self.assertEqual(pattern["merchant"], "Netflix")
        self.assertEqual(pattern["frequency"], "monthly")
        self.assertEqual(pattern["confidence"], "high")
        self.assertEqual(pattern["amount_change_since_first"], 2)

    def test_invalid_input_returns_an_error_instead_of_partial_totals(self):
        bad_row = transaction("08/01/2026", "Cafe", 10, "Dining")

        result = aggregate_monthly_spending([bad_row], "2026-08")

        self.assertEqual(result["status"], "error")
        self.assertIn("YYYY-MM-DD", result["message"])
        self.assertEqual(
            aggregate_monthly_spending(
                [transaction("2026-08-01", "Cafe", 10, "Dining")],
                "August",
            )["status"],
            "error",
        )

    def test_classification_result_shape_can_be_analyzed(self):
        classified = {
            "category": "Transportation",
            "transaction": {
                "transaction_date": "2026-08-01",
                "merchant": "Uber",
                "amount": 20,
                "currency": "USD",
            },
        }

        result = aggregate_monthly_spending([classified], "2026-08")

        self.assertEqual(
            result["totals_by_currency"]["USD"]["Transportation"],
            20,
        )

    def test_get_transactions_combines_all_filters(self):
        rows = [
            transaction(
                "2026-08-20", "Corner Coffee", 8, "Dining", account="Checking"
            ),
            transaction(
                "2026-08-10", "Noodle House", 20, "Dining", account="Checking"
            ),
            transaction(
                "2026-07-30", "Corner Coffee", 7, "Dining", account="Checking"
            ),
            transaction(
                "2026-08-22", "Corner Coffee", 9, "Dining", account="Travel Card"
            ),
        ]

        result = get_transactions(
            rows,
            start_date="2026-08-01",
            end_date="2026-08-31",
            category="dining",
            merchant="coffee",
            account="checking",
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["match_count"], 1)
        self.assertEqual(result["transactions"][0]["merchant"], "Corner Coffee")

    def test_get_transactions_rejects_reversed_date_range(self):
        rows = [
            transaction(
                "2026-08-20", "Corner Coffee", 8, "Dining", account="Checking"
            )
        ]

        result = get_transactions(
            rows,
            start_date="2026-09-01",
            end_date="2026-08-01",
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("on or before", result["message"])

    def test_summarize_accounts_reports_balance_and_monthly_cash_flow(self):
        accounts = [
            {
                "account_name": "Checking",
                "account_type": "checking",
                "current_balance": 5000,
                "currency": "USD",
            },
            {
                "account_name": "Savings",
                "account_type": "savings",
                "current_balance": 12000,
                "currency": "USD",
            },
        ]
        rows = [
            transaction(
                "2026-08-01", "Employer", 3000, "Income", account="Checking"
            ),
            transaction(
                "2026-08-02", "Landlord", 1000, "Housing", account="Checking"
            ),
            transaction(
                "2026-08-03", "Store Refund", -20, "Shopping", account="Checking"
            ),
            transaction(
                "2026-08-04", "Savings Transfer", 500, "Transfers", account="Checking"
            ),
            transaction(
                "2026-08-05", "Cash Cafe", 5, "Dining", account="Cash Wallet"
            ),
        ]

        result = summarize_accounts(accounts, rows, "2026-08")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["unmatched_transaction_count"], 1)
        checking = result["accounts"][0]
        self.assertEqual(checking["current_balance"], 5000)
        self.assertEqual(checking["monthly_income"], 3000)
        self.assertEqual(checking["monthly_spending"], 980)
        self.assertEqual(checking["monthly_net_cash_flow"], 2020)
        self.assertEqual(checking["excluded_transfer_count"], 1)

    def test_summarize_accounts_rejects_currency_mismatch(self):
        accounts = [
            {
                "account_name": "Checking",
                "account_type": "checking",
                "current_balance": 5000,
                "currency": "USD",
            }
        ]
        rows = [
            transaction(
                "2026-08-01",
                "Paris Cafe",
                10,
                "Dining",
                currency="EUR",
                account="Checking",
            )
        ]

        result = summarize_accounts(accounts, rows, "2026-08")

        self.assertEqual(result["status"], "error")
        self.assertIn("Currency mismatch", result["message"])


if __name__ == "__main__":
    unittest.main()
