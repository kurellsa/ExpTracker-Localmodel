"""
Agent 1: Classifier.

Generative classifier using a small Llama model via Ollama. Produces a
Schedule C category, confidence level, and free-text reasoning.
"""
from __future__ import annotations

import json
import os
from typing import Any

import ollama

from app.models import get_all_categories

CLASSIFIER_MODEL = os.getenv("CATEGORIZER_MODEL", "llama3.2:3b")

_SYSTEM_PROMPT = """You are a US CPA specializing in S-Corporation tax returns.
Your job is to categorize business transactions into IRS Schedule C categories.
Always respond with valid JSON only — no explanation, no markdown fences."""


def _build_prompt(description: str, amount: float, categories: list[str]) -> str:
    categories_list = "\n".join(f"- {c}" for c in categories)
    return f"""Categorize this S-Corp business transaction into exactly one Schedule C category.

Available categories:
{categories_list}

Transaction:
  Description: {description}
  Amount: ${amount:.2f}

Rules:
- Airlines/airfare/hotels/lodging/rental cars/Uber/Lyft → "Travel"
- Timeshare/vacation ownership/resort fees → "PERSONAL (excluded)"
- Restaurant/food/dining/DoorDash/UberEats → "Meals (50% deductible)"
- Gas stations for business vehicle → "Car & Truck (Actual)"
- Laptops, computers, phones >$2,500 → "Depreciation / Section 179"
- Office supplies, small equipment → "Supplies"
- Software subscriptions → "Office Expense"
- If clearly personal (grocery store, gym, clothing) → "PERSONAL (excluded)"
- Uncertain → use "Other Business Expense" with low confidence

Respond with JSON only:
{{"category": "<exact category name>", "confidence": "high|medium|low", "reasoning": "<one sentence>"}}"""


def _parse_response(raw: str, categories: list[str]) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    result = json.loads(raw.strip())
    if result.get("category") not in categories:
        result["category"] = "Other Business Expense"
        result["confidence"] = "low"
    return result


def classify(description: str, amount: float) -> dict[str, Any]:
    """Run the classifier agent. Returns {category, confidence, reasoning}."""
    try:
        categories = get_all_categories()
        response = ollama.chat(
            model=CLASSIFIER_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_prompt(description, amount, categories)},
            ],
            options={"temperature": 0.1},
        )
        return _parse_response(response["message"]["content"], categories)
    except Exception as e:
        return {
            "category": "Other Business Expense",
            "confidence": "low",
            "reasoning": f"Classifier error: {e}",
        }
