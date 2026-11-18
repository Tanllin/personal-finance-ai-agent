"""Deterministic local transaction tools used by Payment Planner."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
import math
import re
from statistics import median
from typing import Any

from agents import function_tool
from pydantic import BaseModel, ConfigDict


CATEGORY_RULES = {
    "Income": ("payroll", "salary", "direct deposit", "paycheck"),
    "Transfers": ("zelle", "venmo", "cash app", "paypal transfer", "bank transfer"),
    "Housing": ("rent", "mortgage", "property management", "hoa"),
    "Utilities": (
        "electric",
        "water bill",
        "gas bill",
        "internet",
        "comcast",
        "xfinity",
        "at&t",
        "verizon",
        "t-mobile",
    ),
    "Groceries": (
        "whole foods",
        "trader joe",
        "kroger",
        "aldi",
        "costco",
        "walmart grocery",
        "instacart",
        "supermarket",
        "grocery",
    ),
    "Dining": (
        "restaurant",
        "cafe",
        "coffee",
        "starbucks",
        "mcdonald",
        "doordash",
        "ubereats",
        "uber eats",
        "grubhub",
    ),
    "Transportation": (
        "uber",
        "lyft",
        "taxi",
        "transit",
        "metro",
        "parking",
        "shell",
        "chevron",
        "exxon",
    ),
    "Shopping": ("amazon", "target", "walmart", "best buy", "ikea", "etsy"),
    "Entertainment": (
        "netflix",
        "spotify",
        "hulu",
        "disney+",
        "cinema",
        "movie theater",
    ),
    "Health": (
        "pharmacy",
        "cvs",
        "walgreens",
        "hospital",
        "clinic",
        "dentist",
        "medical",
    ),
    "Travel": ("airlines", "airbnb", "hotel", "marriott", "hilton", "expedia"),
    "Fees": ("service fee", "late fee", "overdraft", "atm fee", "bank fee"),
}

CURRENCY_BY_SYMBOL = {"$": "USD", "€": "EUR", "£": "GBP"}

SENSITIVE_INPUT_PATTERN = re.compile(
    r"\b(?:password|passcode|pin|routing number|account number|cvv|security code|"
    r"authentication token|security answer)\b",
    re.IGNORECASE,
)

POSSIBLE_FULL_CARD_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")

NON_SPENDING_CATEGORIES = {"income", "transfers"}


class AnalyticsTransaction(BaseModel):
    """Minimal normalized record accepted by the analytics Agent tools."""

    model_config = ConfigDict(extra="forbid")

    transaction_date: str
    merchant: str
    amount: float
    currency: str
    category: str


class LedgerTransaction(AnalyticsTransaction):
    """Analytics transaction associated with a non-sensitive account label."""

    account: str


class AccountRecord(BaseModel):
    """User-provided account summary without account or routing numbers."""

    model_config = ConfigDict(extra="forbid")

    account_name: str
    account_type: str
    current_balance: float
    currency: str


def _merchant_from_description(description: str) -> str | None:
    """Remove common bank-statement noise from a merchant description."""

    merchant = re.sub(
        r"\b(?:pending|pos|purchase|debit card|credit card|card)\b",
        " ",
        description,
        flags=re.IGNORECASE,
    )
    merchant = re.split(r"\s{2,}|\s+[\*#]\s*|\*", merchant, maxsplit=1)[0]
    merchant = re.sub(r"[^\w&'.+\- ]", " ", merchant)
    merchant = re.sub(r"\s+", " ", merchant).strip(" -")
    return merchant.title() if merchant else None


def parse_transaction(user_input: str) -> dict:
    """Parse one raw transaction line into structured fields.

    Use this whenever the user supplies transaction text such as
    ``UBER *TRIP $20.00``. The tool rejects likely credentials before returning
    any transaction content.

    Args:
        user_input: One raw transaction description supplied by the user.

    Returns:
        dict: Status, description, merchant, amount, currency, optional date,
        and an optional message.
    """

    raw_input = (user_input or "").strip()
    if not raw_input:
        return {
            "status": "error",
            "raw_input": user_input,
            "description": None,
            "merchant": None,
            "amount": None,
            "currency": None,
            "transaction_date": None,
            "message": "No transaction text was provided.",
        }

    if (
        SENSITIVE_INPUT_PATTERN.search(raw_input)
        or POSSIBLE_FULL_CARD_PATTERN.search(raw_input)
    ):
        return {
            "status": "error",
            "raw_input": "[redacted]",
            "description": None,
            "merchant": None,
            "amount": None,
            "currency": None,
            "transaction_date": None,
            "message": (
                "Remove banking credentials or authentication information "
                "before parsing."
            ),
        }

    date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", raw_input)
    transaction_date = date_match.group(1) if date_match else None

    amount_match = re.search(
        r"(?P<symbol>[$€£])\s*(?P<signed>[-+]?\(?\d[\d,]*(?:\.\d{1,2})?\)?)"
        r"|(?P<coded>[-+]?\(?\d[\d,]*(?:\.\d{1,2})?\)?)\s*"
        r"(?P<code>USD|EUR|GBP|CAD|AUD)\b",
        raw_input,
        re.IGNORECASE,
    )
    if not amount_match:
        amount_match = re.search(
            r"(?P<plain>[-+]?\(?\d[\d,]*\.\d{1,2}\)?)\s*$",
            raw_input,
        )

    amount = None
    currency = None
    description = raw_input
    if amount_match:
        amount_text = next(
            value
            for key in ("signed", "coded", "plain")
            if (value := amount_match.groupdict().get(key)) is not None
        )
        is_parenthesized = amount_text.startswith("(") and amount_text.endswith(")")
        amount = float(amount_text.strip("()").replace(",", ""))
        if is_parenthesized:
            amount = -abs(amount)

        symbol = amount_match.groupdict().get("symbol")
        code = amount_match.groupdict().get("code")
        currency = (
            CURRENCY_BY_SYMBOL.get(symbol)
            if symbol
            else (code.upper() if code else None)
        )
        description = (
            raw_input[: amount_match.start()] + raw_input[amount_match.end() :]
        ).strip()

    if date_match:
        description = description.replace(date_match.group(1), " ")
    description = re.sub(r"\s+", " ", description).strip(" -,")

    return {
        "status": "ok",
        "raw_input": raw_input,
        "description": description or None,
        "merchant": _merchant_from_description(description),
        "amount": amount,
        "currency": currency,
        "transaction_date": transaction_date,
        "message": (
            None
            if amount is not None
            else "No amount was found in the transaction text."
        ),
    }


def categorize_transaction(description: str, merchant: str | None = None) -> dict:
    """Categorize a parsed transaction with deterministic local rules.

    Call this after ``parse_transaction`` and pass through the returned merchant
    and description. Unknown transactions return ``Other`` instead of a guess.

    Args:
        description: Parsed transaction description, such as ``UBER *TRIP``.
        merchant: Optional normalized merchant returned by ``parse_transaction``.

    Returns:
        dict: Status, merchant, category, matched keyword, confidence, and message.
    """

    combined_text = " ".join(
        part for part in (merchant, description) if part
    ).strip()
    resolved_merchant = merchant or _merchant_from_description(description or "")

    if not combined_text:
        return {
            "status": "error",
            "merchant": resolved_merchant,
            "category": "Other",
            "matched_keyword": None,
            "confidence": 0.0,
            "message": "A merchant or transaction description is required.",
        }

    normalized = re.sub(r"[^a-z0-9+& ]", " ", combined_text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    matches = [
        (keyword, category)
        for category, keywords in CATEGORY_RULES.items()
        for keyword in keywords
        if keyword in normalized
    ]

    if matches:
        matched_keyword, category = max(matches, key=lambda match: len(match[0]))
        return {
            "status": "ok",
            "merchant": resolved_merchant,
            "category": category,
            "matched_keyword": matched_keyword,
            "confidence": 0.95,
            "message": None,
        }

    return {
        "status": "ok",
        "merchant": resolved_merchant,
        "category": "Other",
        "matched_keyword": None,
        "confidence": 0.0,
        "message": "No local categorization rule matched this transaction.",
    }


def _month_start(month: str) -> date:
    """Validate YYYY-MM and return its first day."""

    if not re.fullmatch(r"\d{4}-(?:0[1-9]|1[0-2])", month or ""):
        raise ValueError("Month must use YYYY-MM format, such as 2026-08.")
    return datetime.strptime(month, "%Y-%m").date()


def _normalize_analytics_transaction(
    transaction: AnalyticsTransaction | dict[str, Any],
    index: int,
) -> dict[str, Any]:
    """Normalize one flat record or a result returned by classify_transaction."""

    if isinstance(transaction, AnalyticsTransaction):
        payload = transaction.model_dump()
    elif isinstance(transaction, dict):
        payload = dict(transaction)
        if isinstance(payload.get("transaction"), dict):
            nested = dict(payload["transaction"])
            nested.setdefault("category", payload.get("category"))
            payload = nested
    else:
        raise ValueError(f"Transaction {index} must be an object.")

    transaction_date = str(
        payload.get("transaction_date") or payload.get("date") or ""
    ).strip()
    try:
        parsed_date = date.fromisoformat(transaction_date)
    except ValueError as error:
        raise ValueError(
            f"Transaction {index} needs a valid YYYY-MM-DD transaction_date."
        ) from error

    merchant = str(payload.get("merchant") or "").strip()
    category = str(payload.get("category") or "").strip()
    currency = str(payload.get("currency") or "").strip().upper()
    if not merchant:
        raise ValueError(f"Transaction {index} needs a merchant.")
    if not category:
        raise ValueError(f"Transaction {index} needs a category.")
    if not re.fullmatch(r"[A-Z]{3}", currency):
        raise ValueError(
            f"Transaction {index} needs a three-letter currency code."
        )
    try:
        amount = float(payload.get("amount"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Transaction {index} needs a numeric amount.") from error
    if not math.isfinite(amount):
        raise ValueError(f"Transaction {index} amount must be finite.")

    return {
        "transaction_date": parsed_date,
        "merchant": merchant,
        "amount": amount,
        "currency": currency,
        "category": category,
    }


def _normalize_analytics_transactions(
    transactions: list[AnalyticsTransaction],
) -> tuple[list[dict[str, Any]], str | None]:
    """Validate a batch atomically so reports never silently omit bad rows."""

    if not transactions:
        return [], "At least one transaction is required."
    normalized = []
    try:
        for index, transaction in enumerate(transactions, start=1):
            normalized.append(_normalize_analytics_transaction(transaction, index))
    except ValueError as error:
        return [], str(error)
    return normalized, None


def _normalize_ledger_transactions(
    transactions: list[LedgerTransaction],
) -> tuple[list[dict[str, Any]], str | None]:
    """Validate analytics fields plus a non-sensitive account label."""

    if not transactions:
        return [], "At least one transaction is required."
    normalized = []
    try:
        for index, transaction in enumerate(transactions, start=1):
            base = _normalize_analytics_transaction(transaction, index)
            if isinstance(transaction, LedgerTransaction):
                account = transaction.account.strip()
            elif isinstance(transaction, dict):
                payload = transaction
                outer_account = payload.get("account")
                if isinstance(payload.get("transaction"), dict):
                    payload = payload["transaction"]
                account = str(payload.get("account") or outer_account or "").strip()
            else:
                account = ""
            if not account:
                raise ValueError(
                    f"Transaction {index} needs a non-sensitive account label."
                )
            normalized.append({**base, "account": account})
    except ValueError as error:
        return [], str(error)
    return normalized, None


def _normalize_accounts(
    accounts: list[AccountRecord],
) -> tuple[list[dict[str, Any]], str | None]:
    """Validate account labels, types, balances, and currencies atomically."""

    if not accounts:
        return [], "At least one account is required."
    normalized = []
    seen_names = set()
    try:
        for index, account in enumerate(accounts, start=1):
            payload = (
                account.model_dump()
                if isinstance(account, AccountRecord)
                else dict(account)
            )
            account_name = str(payload.get("account_name") or "").strip()
            account_type = str(payload.get("account_type") or "").strip()
            currency = str(payload.get("currency") or "").strip().upper()
            if not account_name:
                raise ValueError(f"Account {index} needs an account_name.")
            if account_name.casefold() in seen_names:
                raise ValueError(f"Account names must be unique: {account_name}.")
            seen_names.add(account_name.casefold())
            if not account_type:
                raise ValueError(f"Account {index} needs an account_type.")
            if not re.fullmatch(r"[A-Z]{3}", currency):
                raise ValueError(
                    f"Account {index} needs a three-letter currency code."
                )
            try:
                current_balance = float(payload.get("current_balance"))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Account {index} needs a numeric current_balance."
                ) from error
            if not math.isfinite(current_balance):
                raise ValueError(f"Account {index} balance must be finite.")
            normalized.append(
                {
                    "account_name": account_name,
                    "account_type": account_type,
                    "current_balance": current_balance,
                    "currency": currency,
                }
            )
    except (TypeError, ValueError) as error:
        return [], str(error)
    return normalized, None


def aggregate_monthly_spending(
    transactions: list[AnalyticsTransaction],
    month: str,
) -> dict:
    """Calculate monthly expense totals grouped by currency and category.

    Positive non-income amounts are spending. Negative expense amounts are refunds
    and reduce spending. Income and transfers are reported as excluded rather than
    mixed into expense totals. Different currencies are never added together.

    Args:
        transactions: Categorized records with date, merchant, amount, currency,
            and category.
        month: Calendar month in ``YYYY-MM`` format.
    """

    try:
        target_month = _month_start(month)
    except ValueError as error:
        return {"status": "error", "message": str(error)}
    normalized, error = _normalize_analytics_transactions(transactions)
    if error:
        return {"status": "error", "message": error}

    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    selected_count = 0
    spending_count = 0
    excluded_count = 0
    for transaction in normalized:
        transaction_date = transaction["transaction_date"]
        if (transaction_date.year, transaction_date.month) != (
            target_month.year,
            target_month.month,
        ):
            continue
        selected_count += 1
        if transaction["category"].casefold() in NON_SPENDING_CATEGORIES:
            excluded_count += 1
            continue
        spending_count += 1
        totals[transaction["currency"]][transaction["category"]] += transaction[
            "amount"
        ]

    totals_by_currency = {
        currency: {
            category: round(amount, 2)
            for category, amount in sorted(category_totals.items())
        }
        for currency, category_totals in sorted(totals.items())
    }
    spending_total_by_currency = {
        currency: round(sum(category_totals.values()), 2)
        for currency, category_totals in totals_by_currency.items()
    }
    return {
        "status": "ok",
        "month": month,
        "selected_transaction_count": selected_count,
        "spending_transaction_count": spending_count,
        "excluded_income_or_transfer_count": excluded_count,
        "totals_by_currency": totals_by_currency,
        "spending_total_by_currency": spending_total_by_currency,
    }


def calculate_net_cash_flow(
    transactions: list[AnalyticsTransaction],
    month: str,
) -> dict:
    """Calculate monthly income minus expenses without counting transfers.

    Income-category amounts add to cash flow. Other category amounts are treated as
    expenses and subtracted; a negative expense therefore behaves as a refund.
    Results remain separated by currency.

    Args:
        transactions: Categorized records with date, merchant, amount, currency,
            and category.
        month: Calendar month in ``YYYY-MM`` format.
    """

    try:
        target_month = _month_start(month)
    except ValueError as error:
        return {"status": "error", "message": str(error)}
    normalized, error = _normalize_analytics_transactions(transactions)
    if error:
        return {"status": "error", "message": error}

    cash_flow: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "income": 0.0,
            "expenses": 0.0,
            "net_cash_flow": 0.0,
            "included_transaction_count": 0,
            "excluded_transfer_count": 0,
        }
    )
    for transaction in normalized:
        transaction_date = transaction["transaction_date"]
        if (transaction_date.year, transaction_date.month) != (
            target_month.year,
            target_month.month,
        ):
            continue
        bucket = cash_flow[transaction["currency"]]
        category = transaction["category"].casefold()
        if category == "transfers":
            bucket["excluded_transfer_count"] += 1
            continue
        bucket["included_transaction_count"] += 1
        if category == "income":
            bucket["income"] += transaction["amount"]
        else:
            bucket["expenses"] += transaction["amount"]

    by_currency = {}
    for currency, values in sorted(cash_flow.items()):
        income = float(values["income"])
        expenses = float(values["expenses"])
        by_currency[currency] = {
            "income": round(income, 2),
            "expenses": round(expenses, 2),
            "net_cash_flow": round(income - expenses, 2),
            "included_transaction_count": values["included_transaction_count"],
            "excluded_transfer_count": values["excluded_transfer_count"],
        }
    return {"status": "ok", "month": month, "by_currency": by_currency}


def summarize_accounts(
    accounts: list[AccountRecord],
    transactions: list[LedgerTransaction],
    month: str,
) -> dict:
    """Summarize user-provided balances and monthly activity by account.

    Current balances are reported exactly as supplied. Monthly income, spending,
    and net cash flow are calculated from matching categorized transactions.
    Transfers are excluded from cash flow, and currencies are never combined.

    Args:
        accounts: Account labels, types, current balances, and currencies. Use names
            such as ``Everyday Checking``; never provide account numbers.
        transactions: Categorized transactions carrying the matching account label.
        month: Activity month in ``YYYY-MM`` format.
    """

    try:
        target_month = _month_start(month)
    except ValueError as error:
        return {"status": "error", "message": str(error)}
    normalized_accounts, error = _normalize_accounts(accounts)
    if error:
        return {"status": "error", "message": error}
    normalized_transactions, error = _normalize_ledger_transactions(transactions)
    if error:
        return {"status": "error", "message": error}

    transactions_by_account: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for transaction in normalized_transactions:
        transactions_by_account[transaction["account"].casefold()].append(transaction)

    summaries = []
    known_accounts = {account["account_name"].casefold() for account in normalized_accounts}
    unmatched_count = sum(
        len(rows)
        for account_key, rows in transactions_by_account.items()
        if account_key not in known_accounts
    )
    for account in normalized_accounts:
        income = 0.0
        spending = 0.0
        included_count = 0
        excluded_transfer_count = 0
        account_transactions = transactions_by_account.get(
            account["account_name"].casefold(), []
        )
        for transaction in account_transactions:
            transaction_date = transaction["transaction_date"]
            if (transaction_date.year, transaction_date.month) != (
                target_month.year,
                target_month.month,
            ):
                continue
            if transaction["currency"] != account["currency"]:
                return {
                    "status": "error",
                    "message": (
                        f"Currency mismatch for {account['account_name']}: account uses "
                        f"{account['currency']} but a transaction uses "
                        f"{transaction['currency']}."
                    ),
                }
            category = transaction["category"].casefold()
            if category == "transfers":
                excluded_transfer_count += 1
                continue
            included_count += 1
            if category == "income":
                income += transaction["amount"]
            else:
                spending += transaction["amount"]
        summaries.append(
            {
                **account,
                "current_balance": round(account["current_balance"], 2),
                "monthly_income": round(income, 2),
                "monthly_spending": round(spending, 2),
                "monthly_net_cash_flow": round(income - spending, 2),
                "included_transaction_count": included_count,
                "excluded_transfer_count": excluded_transfer_count,
            }
        )

    return {
        "status": "ok",
        "month": month,
        "account_count": len(summaries),
        "unmatched_transaction_count": unmatched_count,
        "accounts": summaries,
        "message": (
            "Balances are user-provided snapshots; this tool does not connect to banks."
        ),
    }


def get_transactions(
    transactions: list[LedgerTransaction],
    start_date: str | None = None,
    end_date: str | None = None,
    category: str | None = None,
    merchant: str | None = None,
    account: str | None = None,
) -> dict:
    """Retrieve transactions using optional date, category, merchant, and account filters.

    Filters are combined with AND. Category and account use case-insensitive exact
    matching; merchant uses a case-insensitive substring match. Results are newest
    first and come only from the user-provided batch.

    Args:
        transactions: Categorized transactions with non-sensitive account labels.
        start_date: Optional inclusive start date in ``YYYY-MM-DD`` format.
        end_date: Optional inclusive end date in ``YYYY-MM-DD`` format.
        category: Optional exact category, such as ``Dining``.
        merchant: Optional merchant text, such as ``coffee``.
        account: Optional exact account label, such as ``Everyday Checking``.
    """

    normalized, error = _normalize_ledger_transactions(transactions)
    if error:
        return {"status": "error", "message": error}

    def parse_filter_date(value: str | None, label: str) -> date | None:
        if value is None or not value.strip():
            return None
        try:
            return date.fromisoformat(value.strip())
        except ValueError as date_error:
            raise ValueError(f"{label} must use YYYY-MM-DD format.") from date_error

    try:
        parsed_start = parse_filter_date(start_date, "start_date")
        parsed_end = parse_filter_date(end_date, "end_date")
    except ValueError as error:
        return {"status": "error", "message": str(error)}
    if parsed_start and parsed_end and parsed_start > parsed_end:
        return {
            "status": "error",
            "message": "start_date must be on or before end_date.",
        }

    category_filter = (category or "").strip().casefold()
    merchant_filter = (merchant or "").strip().casefold()
    account_filter = (account or "").strip().casefold()
    matches = []
    for transaction in normalized:
        transaction_date = transaction["transaction_date"]
        if parsed_start and transaction_date < parsed_start:
            continue
        if parsed_end and transaction_date > parsed_end:
            continue
        if category_filter and transaction["category"].casefold() != category_filter:
            continue
        if merchant_filter and merchant_filter not in transaction["merchant"].casefold():
            continue
        if account_filter and transaction["account"].casefold() != account_filter:
            continue
        matches.append(
            {
                **transaction,
                "transaction_date": transaction_date.isoformat(),
                "amount": round(transaction["amount"], 2),
            }
        )
    matches.sort(key=lambda row: row["transaction_date"], reverse=True)
    return {
        "status": "ok",
        "filters": {
            "start_date": start_date or None,
            "end_date": end_date or None,
            "category": category or None,
            "merchant": merchant or None,
            "account": account or None,
        },
        "match_count": len(matches),
        "transactions": matches,
        "message": "Results come only from the transaction batch supplied by the user.",
    }


def compare_monthly_spending(
    transactions: list[AnalyticsTransaction],
    current_month: str,
    previous_month: str,
) -> dict:
    """Compare category spending between two months.

    Percentage change is null when the earlier amount is zero because no finite
    percentage baseline exists. Results are separated by currency.

    Args:
        transactions: Categorized records with date, merchant, amount, currency,
            and category.
        current_month: Newer month in ``YYYY-MM`` format.
        previous_month: Baseline month in ``YYYY-MM`` format.
    """

    if current_month == previous_month:
        return {
            "status": "error",
            "message": "Current and previous month must be different.",
        }
    current = aggregate_monthly_spending(transactions, current_month)
    if current["status"] == "error":
        return current
    previous = aggregate_monthly_spending(transactions, previous_month)
    if previous["status"] == "error":
        return previous

    current_totals = current["totals_by_currency"]
    previous_totals = previous["totals_by_currency"]
    currencies = sorted(set(current_totals) | set(previous_totals))
    comparison = {}
    for currency in currencies:
        current_categories = current_totals.get(currency, {})
        previous_categories = previous_totals.get(currency, {})
        categories = sorted(set(current_categories) | set(previous_categories))
        category_rows = {}
        for category in categories:
            current_amount = float(current_categories.get(category, 0.0))
            previous_amount = float(previous_categories.get(category, 0.0))
            change = current_amount - previous_amount
            if previous_amount == 0:
                percent_change = None
                trend = "new" if current_amount != 0 else "unchanged"
            else:
                percent_change = round(change / abs(previous_amount) * 100, 2)
                if current_amount == 0:
                    trend = "stopped"
                elif change > 0:
                    trend = "increased"
                elif change < 0:
                    trend = "decreased"
                else:
                    trend = "unchanged"
            category_rows[category] = {
                "previous": round(previous_amount, 2),
                "current": round(current_amount, 2),
                "change": round(change, 2),
                "percent_change": percent_change,
                "trend": trend,
            }
        comparison[currency] = category_rows

    return {
        "status": "ok",
        "current_month": current_month,
        "previous_month": previous_month,
        "comparison_by_currency": comparison,
    }


def _recurrence_frequency(days: float) -> tuple[str | None, float]:
    """Map a median interval to a common cadence and tolerance."""

    cadences = [
        ("daily", 1, 1),
        ("weekly", 7, 2),
        ("biweekly", 14, 2),
        ("monthly", 30, 5),
        ("quarterly", 91, 11),
        ("annual", 365, 20),
    ]
    for name, expected_days, tolerance in cadences:
        if abs(days - expected_days) <= tolerance:
            return name, tolerance
    return None, 0


def detect_recurring_transactions(
    transactions: list[AnalyticsTransaction],
) -> dict:
    """Find merchants appearing on a regular daily-to-annual cadence.

    Two matching occurrences produce a tentative pattern; three or more consistent
    occurrences produce a high-confidence pattern. Transfers are excluded. Amount
    changes are reported rather than used to hide a recurring subscription.

    Args:
        transactions: Categorized records with date, merchant, amount, currency,
            and category.
    """

    normalized, error = _normalize_analytics_transactions(transactions)
    if error:
        return {"status": "error", "message": error}

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for transaction in normalized:
        if transaction["category"].casefold() == "transfers":
            continue
        merchant_key = re.sub(
            r"[^a-z0-9]+", " ", transaction["merchant"].casefold()
        ).strip()
        groups[(merchant_key, transaction["currency"], transaction["category"])].append(
            transaction
        )

    recurring = []
    for (_merchant_key, currency, category), rows in groups.items():
        rows.sort(key=lambda row: row["transaction_date"])
        if len(rows) < 2:
            continue
        intervals = [
            (current["transaction_date"] - previous["transaction_date"]).days
            for previous, current in zip(rows, rows[1:])
        ]
        median_interval = float(median(intervals))
        frequency, tolerance = _recurrence_frequency(median_interval)
        if frequency is None or any(
            abs(interval - median_interval) > tolerance for interval in intervals
        ):
            continue

        amounts = [float(row["amount"]) for row in rows]
        typical_amount = float(median(amounts))
        amount_change = amounts[-1] - amounts[0]
        recurring.append(
            {
                "merchant": rows[-1]["merchant"],
                "category": category,
                "currency": currency,
                "frequency": frequency,
                "confidence": "high" if len(rows) >= 3 else "tentative",
                "occurrence_count": len(rows),
                "first_date": rows[0]["transaction_date"].isoformat(),
                "last_date": rows[-1]["transaction_date"].isoformat(),
                "median_interval_days": round(median_interval, 1),
                "typical_amount": round(typical_amount, 2),
                "latest_amount": round(amounts[-1], 2),
                "amount_change_since_first": round(amount_change, 2),
                "estimated_next_date": (
                    rows[-1]["transaction_date"]
                    + timedelta(days=round(median_interval))
                ).isoformat(),
            }
        )

    recurring.sort(
        key=lambda row: (-row["occurrence_count"], row["merchant"].casefold())
    )
    return {
        "status": "ok",
        "recurring_count": len(recurring),
        "patterns": recurring,
    }


# Expose plain functions as Agent tools while keeping them easy to test.
PARSE_TRANSACTION_TOOL = function_tool(parse_transaction)
CATEGORIZE_TRANSACTION_TOOL = function_tool(categorize_transaction)
AGGREGATE_MONTHLY_SPENDING_TOOL = function_tool(aggregate_monthly_spending)
CALCULATE_NET_CASH_FLOW_TOOL = function_tool(calculate_net_cash_flow)
SUMMARIZE_ACCOUNTS_TOOL = function_tool(summarize_accounts)
GET_TRANSACTIONS_TOOL = function_tool(get_transactions)
COMPARE_MONTHLY_SPENDING_TOOL = function_tool(compare_monthly_spending)
DETECT_RECURRING_TRANSACTIONS_TOOL = function_tool(detect_recurring_transactions)

TRANSACTION_TOOLS = [
    PARSE_TRANSACTION_TOOL,
    CATEGORIZE_TRANSACTION_TOOL,
    AGGREGATE_MONTHLY_SPENDING_TOOL,
    CALCULATE_NET_CASH_FLOW_TOOL,
    SUMMARIZE_ACCOUNTS_TOOL,
    GET_TRANSACTIONS_TOOL,
    COMPARE_MONTHLY_SPENDING_TOOL,
    DETECT_RECURRING_TRANSACTIONS_TOOL,
]
