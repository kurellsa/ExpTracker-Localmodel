"""
Local LLM categorizer using Ollama.
Categorizes transactions into IRS Schedule C line items.
"""
import os
import ollama
from app.models import SCHEDULE_C_CATEGORIES

MODEL = os.getenv("CATEGORIZER_MODEL", "qwen3.5:9b")

_SYSTEM_PROMPT = """You are a US CPA specializing in S-Corporation tax returns.
Your job is to categorize business transactions into IRS Schedule C categories.
Always respond with valid JSON only — no explanation, no markdown fences."""

_CATEGORIES_LIST = "\n".join(f"- {c}" for c in SCHEDULE_C_CATEGORIES)


def categorize_transaction(description: str, amount: float) -> dict:
    """
    Returns: {category, confidence, reasoning}
    confidence: "high" | "medium" | "low"
    """
    prompt = f"""Categorize this S-Corp business transaction into exactly one Schedule C category.

Available categories:
{_CATEGORIES_LIST}

Transaction:
  Description: {description}
  Amount: ${amount:.2f}

Rules:
- Restaurant/food/dining/DoorDash/UberEats → "Meals (50% deductible)"
- Gas stations for business vehicle → "Car & Truck (Actual)"
- Laptops, computers, phones >$2,500 → "Depreciation / Section 179"
- Office supplies, small equipment → "Supplies"
- Software subscriptions → "Office Expense"
- If clearly personal (grocery store, gym, clothing) → "PERSONAL (excluded)"
- Uncertain → use "Other Business Expense" with low confidence

Respond with JSON only:
{{"category": "<exact category name>", "confidence": "high|medium|low", "reasoning": "<one sentence>"}}"""

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.1},
        )
        raw = response["message"]["content"].strip()
        # Strip markdown fences if model adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        import json
        result = json.loads(raw.strip())
        # Validate category is in our list
        if result.get("category") not in SCHEDULE_C_CATEGORIES:
            result["category"] = "Other Business Expense"
            result["confidence"] = "low"
        return result
    except Exception as e:
        return {
            "category": "Other Business Expense",
            "confidence": "low",
            "reasoning": f"LLM error: {e}",
        }


def categorize_batch(transactions: list[dict]) -> list[dict]:
    """Categorize a list of {description, amount} dicts. Returns same list with category fields added."""
    results = []
    for txn in transactions:
        result = categorize_transaction(txn["description"], txn["amount"])
        results.append({**txn, **result})
    return results


def check_ollama() -> bool:
    """Return True if Ollama is running and model is available."""
    try:
        models = ollama.list()
        names = [m.model for m in models.models]
        return any(MODEL in n for n in names)
    except Exception:
        return False
