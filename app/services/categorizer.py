"""
Transaction categorizer — thin wrapper over the multi-agent LangGraph pipeline.

The real work happens in app.services.agents.graph:
  classify (Llama 3.2:3b via Ollama) → review (DeBERTa zero-shot) → resolve

This module preserves the original public API so routers don't need to change:
  - categorize_transaction(description, amount) -> dict
  - categorize_batch(transactions) -> list[dict]
  - check_ollama() -> bool
"""
from __future__ import annotations

import logging
import re
import time

import ollama

from app.services.agents.classifier_agent import CLASSIFIER_MODEL
from app.services.agents.graph import run_categorization

# Kept for backward compatibility with any import that references MODEL.
MODEL = CLASSIFIER_MODEL

logger = logging.getLogger(__name__)


def categorize_transaction(description: str, amount: float) -> dict:
    """
    Run the multi-agent categorization graph on a single transaction.

    Returns:
      {
        category, confidence, reasoning, is_approved,
        llm_category, llm_confidence, llm_reasoning,
      }
    """
    return run_categorization(description, amount)


def _normalize_vendor(description: str) -> str:
    """Normalize vendor name for dedup — strip trailing digits, whitespace, location suffixes."""
    name = description.strip().upper()
    # Remove trailing location info like "ATLANTA GA", "CAROLINA PR"
    name = re.sub(r"\s+[A-Z]{2}\s*$", "", name)
    # Remove trailing numbers (store IDs, transaction refs)
    name = re.sub(r"[\s#*\-]+[\dA-Z]*$", "", name)
    return name.strip()


def categorize_batch(transactions: list[dict]) -> list[dict]:
    """
    Categorize a list of {description, amount} dicts via the LangGraph pipeline.

    Vendor dedup: transactions with the same normalized vendor name are
    categorized once and the result is reused, ensuring consistency and
    cutting LLM calls.
    """
    results: list[dict | None] = [None] * len(transactions)

    # Group by normalized vendor — only categorize one representative per vendor.
    vendor_groups: dict[str, list[int]] = {}
    vendor_representative: dict[str, int] = {}
    for i, txn in enumerate(transactions):
        vendor_key = _normalize_vendor(txn["description"])
        if vendor_key not in vendor_groups:
            vendor_groups[vendor_key] = []
            vendor_representative[vendor_key] = i
        vendor_groups[vendor_key].append(i)

    unique_indices = list(vendor_representative.values())

    t0 = time.perf_counter()
    vendor_results: dict[str, dict] = {}
    for i in unique_indices:
        cat_result = categorize_transaction(
            transactions[i]["description"], transactions[i]["amount"]
        )
        vendor_key = _normalize_vendor(transactions[i]["description"])
        vendor_results[vendor_key] = cat_result

    for vendor_key, indices in vendor_groups.items():
        cat_result = vendor_results[vendor_key]
        for i in indices:
            results[i] = {**transactions[i], **cat_result}

    elapsed = time.perf_counter() - t0
    dedup_hits = len(transactions) - len(unique_indices)
    logger.info(
        "batch_complete | %.2fs | %d txns | %d graph runs | %d vendor dedup",
        elapsed,
        len(transactions),
        len(unique_indices),
        dedup_hits,
    )
    return results


def check_ollama() -> bool:
    """Return True if Ollama is running and the classifier model is available."""
    try:
        models = ollama.list()
        names = [m.model for m in models.models]
        return any(MODEL in n for n in names)
    except Exception:
        return False
