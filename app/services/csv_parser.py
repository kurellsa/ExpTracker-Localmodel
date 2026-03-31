"""
Multi-bank CSV parser for business expense tracking.
Supports: Chase, Bank of America, Capital One, Wells Fargo, Generic.
"""
import csv
import io
from datetime import date, datetime
from typing import Any


def parse_csv(content: bytes, filename: str) -> list[dict[str, Any]]:
    text = content.decode("utf-8-sig", errors="replace")
    bank = _detect_bank(text, filename)
    parser = _PARSERS.get(bank, _parse_generic)
    rows = parser(text)
    for r in rows:
        r["bank"] = bank
    return rows


def _detect_bank(text: str, filename: str) -> str:
    header = text[:400].lower()
    fname = filename.lower()

    if "chase" in fname or ("transaction date" in header and "post date" in header and "category" in header):
        return "Chase"
    if "bank of america" in fname or "bankofamerica" in fname or "bofa" in fname:
        return "Bank of America"
    if "capital one" in fname or "capitalone" in fname:
        return "Capital One"
    if "wells fargo" in fname or "wellsfargo" in fname:
        return "Wells Fargo"
    if "amex" in fname or "american express" in fname:
        return "American Express"
    return "Generic"


def _parse_date(value: str) -> date:
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y", "%m/%d/%y", "%Y/%m/%d"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {value}")


def _to_expense(amount_str: str) -> float | None:
    """Return positive float for expenses, skip credits/income (negative)."""
    try:
        val = float(amount_str.replace("$", "").replace(",", "").strip())
        # expenses are negative in most bank exports; flip to positive
        return round(abs(val), 2) if val < 0 else round(val, 2)
    except ValueError:
        return None


# ── Chase ──────────────────────────────────────────────────────────────────────
# Columns: Transaction Date, Post Date, Description, Category, Type, Amount, Memo
def _parse_chase(text: str) -> list[dict]:
    rows = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        try:
            amount_raw = float(row.get("Amount", "0").replace(",", ""))
            if amount_raw >= 0:
                continue  # skip credits / payments
            rows.append({
                "date": _parse_date(row["Transaction Date"]),
                "description": row.get("Description", "").strip(),
                "amount": round(abs(amount_raw), 2),
                "account": row.get("Type", ""),
            })
        except (KeyError, ValueError):
            continue
    return rows


# ── Bank of America ────────────────────────────────────────────────────────────
# Columns: Date, Description, Amount, Running Bal.
def _parse_bofa(text: str) -> list[dict]:
    rows = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        try:
            amount_raw = float(row.get("Amount", "0").replace(",", ""))
            if amount_raw >= 0:
                continue
            rows.append({
                "date": _parse_date(row["Date"]),
                "description": row.get("Description", "").strip(),
                "amount": round(abs(amount_raw), 2),
                "account": "",
            })
        except (KeyError, ValueError):
            continue
    return rows


# ── Capital One ────────────────────────────────────────────────────────────────
# Columns: Transaction Date, Posted Date, Card No., Description, Category, Debit, Credit
def _parse_capital_one(text: str) -> list[dict]:
    rows = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        try:
            debit = row.get("Debit", "").strip()
            if not debit:
                continue
            rows.append({
                "date": _parse_date(row["Transaction Date"]),
                "description": row.get("Description", "").strip(),
                "amount": round(float(debit.replace(",", "")), 2),
                "account": row.get("Card No.", ""),
            })
        except (KeyError, ValueError):
            continue
    return rows


# ── Wells Fargo ────────────────────────────────────────────────────────────────
# No header row: Date, Amount, *, *, Description
def _parse_wells_fargo(text: str) -> list[dict]:
    rows = []
    reader = csv.reader(io.StringIO(text))
    for row in reader:
        if len(row) < 5:
            continue
        try:
            amount_raw = float(row[1].replace(",", ""))
            if amount_raw >= 0:
                continue
            rows.append({
                "date": _parse_date(row[0]),
                "description": row[4].strip(),
                "amount": round(abs(amount_raw), 2),
                "account": "",
            })
        except (IndexError, ValueError):
            continue
    return rows


# ── American Express ───────────────────────────────────────────────────────────
# Columns: Date, Description, Amount
def _parse_amex(text: str) -> list[dict]:
    rows = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        try:
            amount_raw = float(row.get("Amount", "0").replace(",", ""))
            if amount_raw <= 0:
                continue  # Amex: positive = charge
            rows.append({
                "date": _parse_date(row["Date"]),
                "description": row.get("Description", "").strip(),
                "amount": round(amount_raw, 2),
                "account": "",
            })
        except (KeyError, ValueError):
            continue
    return rows


# ── Generic fallback ───────────────────────────────────────────────────────────
def _parse_generic(text: str) -> list[dict]:
    rows = []
    reader = csv.DictReader(io.StringIO(text))
    headers = reader.fieldnames or []
    headers_lower = [h.lower() for h in headers]

    date_col = next((headers[i] for i, h in enumerate(headers_lower) if "date" in h), None)
    desc_col = next((headers[i] for i, h in enumerate(headers_lower) if "desc" in h or "memo" in h or "narr" in h), None)
    amount_col = next((headers[i] for i, h in enumerate(headers_lower) if "amount" in h or "debit" in h), None)

    if not all([date_col, desc_col, amount_col]):
        return rows

    for row in reader:
        try:
            amount_str = row.get(amount_col, "0")
            val = float(amount_str.replace("$", "").replace(",", "").strip())
            if val == 0:
                continue
            rows.append({
                "date": _parse_date(row[date_col]),
                "description": row.get(desc_col, "").strip(),
                "amount": round(abs(val), 2),
                "account": "",
            })
        except (KeyError, ValueError):
            continue
    return rows


_PARSERS = {
    "Chase": _parse_chase,
    "Bank of America": _parse_bofa,
    "Capital One": _parse_capital_one,
    "Wells Fargo": _parse_wells_fargo,
    "American Express": _parse_amex,
    "Generic": _parse_generic,
}
