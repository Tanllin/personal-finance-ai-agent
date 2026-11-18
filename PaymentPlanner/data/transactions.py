"""Fixed synthetic, manually labeled examples, shaped like Lesson 08 data/jobs.py.

These are demonstration records, not the user's actual bank history. `verified`
means the label is accepted for this demo corpus, not independently bank-verified.
New queries are never appended to this file or automatically promoted to examples.
"""

# Unique ID, original example description, metadata.
TRANSACTIONS = [
    ("txn_001", "CITY RAIL SINGLE RIDE", {"merchant": "City Rail", "category": "Transportation", "verified": True}),
    ("txn_002", "CITY RAIL WEEKLY PASS", {"merchant": "City Rail", "category": "Transportation", "verified": True}),
    ("txn_003", "CITY RAIL TICKET MACHINE", {"merchant": "City Rail", "category": "Transportation", "verified": True}),
    ("txn_004", "CITY RAIL MONTHLY TICKET", {"merchant": "City Rail", "category": "Transportation", "verified": True}),
    ("txn_005", "CITY RAIL FARE", {"merchant": "City Rail", "category": "Transportation", "verified": True}),
    ("txn_006", "GREEN BASKET FRESH PRODUCE", {"merchant": "Green Basket", "category": "Groceries", "verified": True}),
    ("txn_007", "GREEN BASKET FRUIT AND VEGETABLES", {"merchant": "Green Basket", "category": "Groceries", "verified": True}),
    ("txn_008", "GREEN BASKET FOOD MARKET", {"merchant": "Green Basket", "category": "Groceries", "verified": True}),
    ("txn_009", "GREEN BASKET MILK AND BREAD", {"merchant": "Green Basket", "category": "Groceries", "verified": True}),
    ("txn_010", "GREEN BASKET WEEKLY FOOD SHOP", {"merchant": "Green Basket", "category": "Groceries", "verified": True}),
    ("txn_011", "NOODLE HOUSE LUNCH", {"merchant": "Noodle House", "category": "Dining", "verified": True}),
    ("txn_012", "NOODLE HOUSE DINNER", {"merchant": "Noodle House", "category": "Dining", "verified": True}),
    ("txn_013", "NOODLE HOUSE TAKEOUT", {"merchant": "Noodle House", "category": "Dining", "verified": True}),
    ("txn_014", "NOODLE HOUSE MEAL", {"merchant": "Noodle House", "category": "Dining", "verified": True}),
    ("txn_015", "NOODLE HOUSE TABLE SERVICE", {"merchant": "Noodle House", "category": "Dining", "verified": True}),
    ("txn_016", "STREAMBOX MONTHLY VIDEO PLAN", {"merchant": "StreamBox", "category": "Entertainment", "verified": True}),
    ("txn_017", "STREAMBOX VIDEO SUBSCRIPTION", {"merchant": "StreamBox", "category": "Entertainment", "verified": True}),
    ("txn_018", "STREAMBOX MOVIE RENTAL", {"merchant": "StreamBox", "category": "Entertainment", "verified": True}),
    ("txn_019", "STREAMBOX FILM STREAMING", {"merchant": "StreamBox", "category": "Entertainment", "verified": True}),
    ("txn_020", "STREAMBOX TV MEMBERSHIP", {"merchant": "StreamBox", "category": "Entertainment", "verified": True}),
    ("txn_021", "BRIGHT ENERGY MONTHLY BILL", {"merchant": "Bright Energy", "category": "Utilities", "verified": True}),
    ("txn_022", "BRIGHT ENERGY HOME POWER", {"merchant": "Bright Energy", "category": "Utilities", "verified": True}),
    ("txn_023", "BRIGHT ENERGY RESIDENTIAL BILL", {"merchant": "Bright Energy", "category": "Utilities", "verified": True}),
    ("txn_024", "BRIGHT ENERGY POWER SUPPLY", {"merchant": "Bright Energy", "category": "Utilities", "verified": True}),
    ("txn_025", "BRIGHT ENERGY UTILITY CHARGE", {"merchant": "Bright Energy", "category": "Utilities", "verified": True}),
]


def ids() -> list[str]:
    return [record[0] for record in TRANSACTIONS]


def texts() -> list[str]:
    return [record[1] for record in TRANSACTIONS]


def metas() -> list[dict]:
    return [dict(record[2]) for record in TRANSACTIONS]
