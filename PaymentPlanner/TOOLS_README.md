# Payment Planner Tools Guide

This document explains the behavior of every function in `tools.py` in plain
language. It focuses on what information each function expects, what decisions it
makes, what it returns, and how it handles uncertain or invalid data.

The tools are read-only. They analyze information supplied by the user; they do not
connect to a bank, retrieve live balances, or create or execute payments.

## Transaction and account formats

The analytics tools expect a categorized transaction with these fields:

```text
transaction_date: 2026-09-15
merchant: Uber
amount: 20.00
currency: USD
category: Transportation
```

Tools that work at the account level also expect a non-sensitive account label:

```text
account: Everyday Checking
```

An account summary record contains:

```text
account_name: Everyday Checking
account_type: checking
current_balance: 2450.75
currency: USD
```

Account labels are names chosen by the user. They must not be account numbers.

---

## Public Agent tools

These functions are exposed to the Payment Planner agent.

## 1. `parse_transaction(user_input)`

### Purpose

Turns one raw bank-statement-style transaction into structured information that the
other tools can understand.

For example, the input might be:

`2026-09-15 UBER *TRIP $20.00`

### Process

1. **Clean the input.** It removes unnecessary spaces at the beginning and end. If
   the user did not provide anything, it stops and returns an error.
2. **Check for sensitive information.** Before parsing, it checks for passwords,
   PINs, routing numbers, account numbers, CVVs, security codes, authentication
   information, security answers, and text resembling a full card number. If any of
   these appear, the input is rejected and the original text is replaced with
   `[redacted]` in the result.
3. **Look for a transaction date.** It recognizes a date written as `YYYY-MM-DD`.
   From the example, it extracts `2026-09-15`. If there is no date, the date remains
   empty.
4. **Look for an amount and currency.** It recognizes values such as `$20.00`, `€15`,
   `20 USD`, `20.00 CAD`, and parenthesized amounts. Currency symbols are converted
   into codes: `$` becomes `USD`, `€` becomes `EUR`, and `£` becomes `GBP`.
5. **Convert the amount to a number.** Commas are removed, so `$1,250.50` becomes
   `1250.50`. A parenthesized value such as `($20.00)` becomes `-20.00`.
6. **Remove the extracted date and amount.** This prevents them from becoming part
   of the merchant name. The example becomes approximately `UBER *TRIP`.
7. **Clean the remaining description.** Extra spaces and surrounding punctuation
   are removed.
8. **Extract the merchant.** The cleaned description is passed to
   `_merchant_from_description()`. In the example, the merchant becomes `Uber`.
9. **Return structured data.** The result contains the original input, cleaned
   description, merchant, amount, currency, date, status, and any warning message.

### Conceptual result

```text
status: ok
description: UBER *TRIP
merchant: Uber
amount: 20.00
currency: USD
transaction_date: 2026-09-15
```

If no amount is found, parsing can still succeed, but the result includes a message
explaining that the amount is missing.

---

## 2. `categorize_transaction(description, merchant)`

### Purpose

Assigns a category using deterministic merchant and keyword rules. It does not use a
language model or ChromaDB.

Example input:

```text
description: UBER *TRIP
merchant: Uber
```

### Process

1. **Combine the merchant and description.** This gives the tool more evidence than
   either field alone.
2. **Normalize the text.** Capitalization and irrelevant punctuation are removed so
   keyword matching is consistent.
3. **Check the known category rules.** Categories include Income, Transfers,
   Housing, Utilities, Groceries, Dining, Transportation, Shopping, Entertainment,
   Health, Travel, and Fees.
4. **Choose the most specific matching keyword.** If several keywords match, the
   longest match is preferred. This helps `direct deposit` win over a shorter,
   less-specific match.
5. **Return the category and evidence.** A successful match includes the normalized
   merchant, category, matched keyword, and confidence.
6. **Abstain when nothing matches.** Unknown transactions return `Other` with zero
   confidence instead of receiving an invented category.

### Conceptual result

```text
merchant: Uber
category: Transportation
matched_keyword: uber
confidence: 0.95
```

When the category is `Other`, the agent may use `search_similar_transactions`, which
is implemented in `semantic_search.py`, as the semantic fallback.

---

## 3. `aggregate_monthly_spending(transactions, month)`

### Purpose

Calculates expense totals for one month, grouped by category and currency.

Example month: `2026-09`

Example transactions might include Dining, Groceries, Income, and Transfers records.

### Process

1. **Validate the month.** The month must use `YYYY-MM` format and contain a real
   month number from `01` through `12`.
2. **Validate the complete transaction batch.** Every row must have a valid date,
   merchant, numeric amount, three-letter currency, and category. If one row is
   invalid, the tool returns an error instead of silently creating partial totals.
3. **Keep only the selected month.** Transactions from other months are ignored.
4. **Exclude non-spending categories.** Income and Transfers are counted as excluded
   records rather than spending.
5. **Group expenses by currency and category.** USD and EUR values remain separate;
   the tool never adds unlike currencies together.
6. **Handle refunds.** A negative expense reduces that category's total. For example,
   a `$50` purchase and a `-$10` refund result in `$40` of net spending.
7. **Return totals and counts.** The result includes selected record count, spending
   record count, excluded Income/Transfer count, category totals, and overall spending
   totals for each currency.

### Conceptual result

```text
month: 2026-09
totals_by_currency:
  USD:
    Dining: 120.00
    Groceries: 350.00
spending_total_by_currency:
  USD: 470.00
```

---

## 4. `calculate_net_cash_flow(transactions, month)`

### Purpose

Calculates monthly income minus expenses while excluding transfers.

### Process

1. **Validate the requested month and every transaction.** Invalid data stops the
   calculation so the result cannot silently omit a row.
2. **Keep transactions from the selected month.** Other months are ignored.
3. **Separate income.** Transactions categorized as Income are added to the income
   total.
4. **Separate expenses.** All categories other than Income and Transfers are added to
   the expense total.
5. **Exclude transfers.** Moving money between the user's accounts is not treated as
   new income or spending.
6. **Apply refunds naturally.** A negative expense reduces total expenses and
   therefore increases net cash flow.
7. **Calculate net cash flow.** The formula is `income - expenses`.
8. **Keep currencies separate.** Each currency receives an independent result.

### Example

If the month contains `$3,000` of income, `$1,200` of expenses, and a `-$50` refund:

```text
income: 3000.00
expenses: 1150.00
net_cash_flow: 1850.00
```

The result also reports included transactions and excluded transfers.

---

## 5. `summarize_accounts(accounts, transactions, month)`

### Purpose

Creates an account-by-account summary using balances and transaction records supplied
by the user.

This is not a live-bank tool. The current balance is a snapshot provided by the user.

### Process

1. **Validate the month.** It must use `YYYY-MM` format.
2. **Validate the accounts.** Each account needs a unique non-sensitive name, an
   account type, a finite numeric balance, and a three-letter currency.
3. **Validate the transaction batch.** Each transaction needs the normal analytics
   fields plus an account label.
4. **Match transactions to accounts.** Matching is case-insensitive, so `Checking`
   and `checking` refer to the same account label.
5. **Keep monthly activity only.** Transactions outside the requested month do not
   affect that month's income, spending, or cash flow.
6. **Check currency consistency.** A USD account cannot silently include a EUR
   transaction. A mismatch returns an error.
7. **Calculate account-level income and spending.** Income is added separately;
   other non-transfer categories are treated as spending.
8. **Exclude transfers.** Transfers are counted but do not affect cash flow.
9. **Calculate account-level net cash flow.** For every account, the tool subtracts
   monthly spending from monthly income.
10. **Report unmatched transactions.** Transactions referencing an account not in the
    supplied account list are not silently assigned to another account.

### Conceptual result

```text
account_name: Everyday Checking
account_type: checking
current_balance: 2450.75
currency: USD
monthly_income: 3000.00
monthly_spending: 1800.00
monthly_net_cash_flow: 1200.00
```

---

## 6. `get_transactions(transactions, start_date, end_date, category, merchant, account)`

### Purpose

Retrieves records from a transaction batch supplied by the user using optional
filters.

It does not connect to a bank or load a private ledger automatically.

### Process

1. **Validate every transaction.** Each row must include a date, merchant, amount,
   currency, category, and account label.
2. **Validate the date filters.** Dates must use `YYYY-MM-DD`. If both dates are
   provided, the start date cannot be later than the end date.
3. **Apply the date range.** Start and end dates are inclusive.
4. **Apply the category filter.** Category matching is case-insensitive and exact.
   `Dining` matches `dining`, but it does not match `Dining and Travel`.
5. **Apply the merchant filter.** Merchant matching is case-insensitive and partial.
   Searching for `coffee` can match `Corner Coffee`.
6. **Apply the account filter.** Account matching is case-insensitive and exact.
7. **Combine filters using AND.** A row must satisfy every supplied filter.
8. **Sort the results.** Matching transactions are returned newest first.
9. **Return the applied filters and match count.** This makes the result easy to
   explain and audit.

Any filter can be left empty. For example, providing only `category: Dining` returns
all Dining transactions in the supplied batch.

---

## 7. `compare_monthly_spending(transactions, current_month, previous_month)`

### Purpose

Compares category spending between two different months.

### Process

1. **Require two different months.** Comparing a month with itself returns an error.
2. **Aggregate both months.** The function uses the same rules as
   `aggregate_monthly_spending`, including Income/Transfer exclusion and separate
   currencies.
3. **Create the complete category set.** A category is included even if it appears in
   only one of the two months.
4. **Calculate the amount change.** This is `current amount - previous amount`.
5. **Calculate percentage change.** When the previous amount is nonzero, the change
   is divided by the absolute previous amount.
6. **Avoid misleading division by zero.** If the previous amount was zero, percentage
   change is left empty rather than reported as infinity.
7. **Assign a trend.** Each category is marked as `increased`, `decreased`,
   `unchanged`, `new`, or `stopped`.

### Example

If Dining was `$100` in August and `$120` in September:

```text
previous: 100.00
current: 120.00
change: 20.00
percent_change: 20.00
trend: increased
```

---

## 8. `detect_recurring_transactions(transactions)`

### Purpose

Finds merchants that appear repeatedly on a recognizable schedule.

### Process

1. **Validate every transaction.** Invalid rows stop the analysis instead of being
   silently discarded.
2. **Exclude transfers.** Repeated transfers between the user's own accounts are not
   presented as subscriptions or recurring spending.
3. **Group similar records.** Transactions are grouped by normalized merchant,
   currency, and category. This keeps a USD Netflix charge separate from a EUR charge
   or a differently categorized merchant.
4. **Sort each group by date.** The tool determines the number of days between every
   neighboring occurrence.
5. **Require at least two occurrences.** One transaction is not enough to establish a
   pattern.
6. **Find the typical interval.** The median interval prevents one unusual date from
   having too much influence.
7. **Recognize common schedules.** Supported patterns are daily, weekly, biweekly,
   monthly, quarterly, and annual, with reasonable date tolerances.
8. **Reject inconsistent timing.** A merchant appearing repeatedly but at irregular
   intervals is not automatically labeled recurring.
9. **Summarize the amounts.** The result includes typical amount, latest amount, and
   amount change since the first occurrence. A price increase does not hide an
   otherwise regular subscription.
10. **Estimate the next date.** The latest date plus the median interval produces an
    approximate next occurrence.
11. **Assign confidence.** Two consistent occurrences are `tentative`; three or more
    are `high` confidence.

### Conceptual result

```text
merchant: Netflix
category: Entertainment
frequency: monthly
confidence: high
occurrence_count: 3
typical_amount: 15.49
estimated_next_date: 2026-10-15
```

---

## Private helper functions

These functions support the public tools. The agent does not call them directly.

## 9. `_merchant_from_description(description)`

### Purpose

Creates a cleaner merchant name from a noisy statement description.

### Process

1. Removes common statement noise such as `pending`, `POS`, `purchase`, `debit card`,
   and `credit card`.
2. Stops at separators commonly used between a merchant and transaction details,
   including `*` and `#`.
3. Removes irrelevant punctuation while preserving useful characters in names.
4. Collapses repeated spaces.
5. Converts the result into title-style capitalization.
6. Returns an empty value if no merchant text remains.

For example, `PENDING POS UBER *TRIP` becomes approximately `Uber`.

---

## 10. `_month_start(month)`

### Purpose

Validates a month used by monthly analytics tools.

### Process

1. Requires the exact `YYYY-MM` shape.
2. Ensures the month is between `01` and `12`.
3. Converts the value into the first calendar day of that month so date comparisons
   are reliable.
4. Raises a clear validation error for values such as `September 2026`, `2026-9`, or
   `2026-13`.

---

## 11. `_normalize_analytics_transaction(transaction, index)`

### Purpose

Converts one transaction into the consistent internal format used by analytics.

### Process

1. Accepts a normal analytics record or the nested result produced by the semantic
   classifier.
2. Reads either `transaction_date` or the shorter `date` field.
3. Requires a real ISO date, nonempty merchant, nonempty category, and three-letter
   currency.
4. Converts the amount into a finite number and rejects missing, nonnumeric, infinite,
   or `NaN` values.
5. Converts currency codes to uppercase.
6. Returns a consistent internal record with a real date value.
7. Includes the row number in errors so the user can locate the invalid transaction.

---

## 12. `_normalize_analytics_transactions(transactions)`

### Purpose

Validates a complete batch of transactions before a report is calculated.

### Process

1. Rejects an empty batch.
2. Sends each row through `_normalize_analytics_transaction()`.
3. Stops if any row is invalid.
4. Returns either the complete normalized batch or one clear error message.

This all-or-nothing behavior prevents totals from looking correct while silently
missing malformed rows.

---

## 13. `_normalize_ledger_transactions(transactions)`

### Purpose

Prepares transactions for account-based tools.

### Process

1. Performs all normal analytics validation.
2. Requires each row to contain a non-sensitive account label.
3. Supports both flat transaction records and nested classifier results.
4. Preserves the account label alongside the normalized transaction.
5. Rejects the entire batch when an account label or required transaction field is
   missing.

---

## 14. `_normalize_accounts(accounts)`

### Purpose

Validates user-provided account summaries before account-level calculations.

### Process

1. Rejects an empty account list.
2. Requires an account name, account type, numeric balance, and three-letter currency.
3. Treats account names case-insensitively when checking uniqueness.
4. Rejects duplicate account names so transactions cannot be assigned ambiguously.
5. Converts balances to finite numbers and currency codes to uppercase.
6. Returns either the complete normalized account list or a clear error.

The account name should be a label such as `Everyday Checking`, never a real account
number.

---

## 15. `_recurrence_frequency(days)`

### Purpose

Translates a typical number of days between transactions into a familiar schedule.

### Process

1. Receives the median interval calculated by `detect_recurring_transactions()`.
2. Compares it with common schedules: daily, weekly, biweekly, monthly, quarterly,
   and annual.
3. Uses a tolerance around each schedule because billing dates can shift due to month
   length, weekends, and holidays.
4. Returns the schedule name and allowed tolerance when a match exists.
5. Returns no schedule when the interval is not close enough to a supported cadence.

---

## Agent tool registration

At the bottom of `tools.py`, the eight public functions are wrapped as OpenAI Agent
function tools and collected in `TRANSACTION_TOOLS`:

1. `parse_transaction`
2. `categorize_transaction`
3. `aggregate_monthly_spending`
4. `calculate_net_cash_flow`
5. `summarize_accounts`
6. `get_transactions`
7. `compare_monthly_spending`
8. `detect_recurring_transactions`

`agent.py` adds three more tools from other modules:

- `classify_payment_transaction`
- `search_similar_transactions`
- `save_planner_preference`

Those functions are documented in their respective modules rather than this
`tools.py` guide.
