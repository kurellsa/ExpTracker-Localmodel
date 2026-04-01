"""
Local LLM categorizer using Ollama.
Categorizes transactions into IRS Schedule C line items.
"""
import os
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import ollama
from app.models import SCHEDULE_C_CATEGORIES

MODEL = os.getenv("CATEGORIZER_MODEL", "qwen2.5:7b")
MAX_WORKERS = int(os.getenv("CATEGORIZER_WORKERS", "4"))

logger = logging.getLogger(__name__)

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

    t0 = time.perf_counter()
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.1},
        )
        elapsed = time.perf_counter() - t0
        raw = response["message"]["content"].strip()

        # Strip markdown fences if model adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw.strip())

        # Validate category is in our list
        if result.get("category") not in SCHEDULE_C_CATEGORIES:
            result["category"] = "Other Business Expense"
            result["confidence"] = "low"

        logger.info("categorized | %.2fs | %s | %s | conf=%s",
                    elapsed, result["category"], description[:60], result.get("confidence"))
        return result

    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.warning("categorize_error | %.2fs | %s | %s", elapsed, description[:60], e)
        return {
            "category": "Other Business Expense",
            "confidence": "low",
            "reasoning": f"LLM error: {e}",
        }


def _normalize_vendor(description: str) -> str:
    """Normalize vendor name for cache lookup — strip trailing digits, whitespace, location suffixes."""
    import re
    name = description.strip().upper()
    # Remove trailing location info like "ATLANTA GA", "CAROLINA PR", city/state patterns
    name = re.sub(r'\s+[A-Z]{2}\s*$', '', name)
    # Remove trailing numbers (store IDs, transaction refs)
    name = re.sub(r'[\s#*\-]+[\dA-Z]*$', '', name)
    return name.strip()


def categorize_batch(transactions: list[dict]) -> list[dict]:
    """Categorize a list of {description, amount} dicts in parallel, with vendor dedup for consistency."""
    results = [None] * len(transactions)

    # Step 1: Group by normalized vendor — only categorize one per unique vendor
    vendor_groups: dict[str, list[int]] = {}
    vendor_representative: dict[str, int] = {}  # vendor_key -> first index
    for i, txn in enumerate(transactions):
        vendor_key = _normalize_vendor(txn["description"])
        if vendor_key not in vendor_groups:
            vendor_groups[vendor_key] = []
            vendor_representative[vendor_key] = i
        vendor_groups[vendor_key].append(i)

    unique_indices = list(vendor_representative.values())

    # Step 2: Parallel LLM calls for unique vendors only
    t0 = time.perf_counter()
    vendor_results: dict[str, dict] = {}
    if unique_indices:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(categorize_transaction, transactions[i]["description"], transactions[i]["amount"]): i
                for i in unique_indices
            }
            for future in as_completed(futures):
                i = futures[future]
                cat_result = future.result()
                vendor_key = _normalize_vendor(transactions[i]["description"])
                vendor_results[vendor_key] = cat_result

    # Step 3: Spread results to all transactions with the same vendor
    for vendor_key, indices in vendor_groups.items():
        cat_result = vendor_results[vendor_key]
        for i in indices:
            results[i] = {**transactions[i], **cat_result}

    elapsed = time.perf_counter() - t0
    cache_hits = len(transactions) - len(unique_indices)
    logger.info("batch_complete | %.2fs | %d txns | %d LLM calls | %d vendor dedup",
                elapsed, len(transactions), len(unique_indices), cache_hits)
    return results


def check_ollama() -> bool:
    """Return True if Ollama is running and model is available."""
    try:
        models = ollama.list()
        names = [m.model for m in models.models]
        return any(MODEL in n for n in names)
    except Exception:
        return False
