"""
Agent 2: Reviewer.

Discriminative second opinion using a DeBERTa-v3 zero-shot NLI classifier
from HuggingFace. Runs locally, different architecture family than the Llama
classifier so errors are uncorrelated.

Device selection:
  - Auto-detects MPS (Apple Silicon) → CUDA → CPU.
  - Override with REVIEWER_DEVICE env var (mps / cuda / cpu).
  - Falls back to CPU if the model fails to load on the selected device.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

from app.models import get_all_categories

logger = logging.getLogger(__name__)

REVIEWER_MODEL = os.getenv(
    "REVIEWER_MODEL", "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
)
REVIEWER_DEVICE_OVERRIDE = os.getenv("REVIEWER_DEVICE", "").strip().lower()

_HYPOTHESIS_TEMPLATE = "This business expense is for {}."


def _select_device() -> str:
    """Pick best available torch device: mps → cuda → cpu."""
    if REVIEWER_DEVICE_OVERRIDE in {"mps", "cuda", "cpu"}:
        return REVIEWER_DEVICE_OVERRIDE
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


@lru_cache(maxsize=1)
def _get_pipeline():
    """Load the zero-shot pipeline once. Falls back to CPU on MPS failure."""
    from transformers import pipeline

    device = _select_device()
    try:
        pipe = pipeline(
            "zero-shot-classification",
            model=REVIEWER_MODEL,
            device=device,
        )
        logger.info("reviewer device: %s (model=%s)", device, REVIEWER_MODEL)
        return pipe
    except Exception as e:
        if device != "cpu":
            logger.warning(
                "reviewer failed on device=%s (%s); falling back to CPU", device, e
            )
            pipe = pipeline(
                "zero-shot-classification",
                model=REVIEWER_MODEL,
                device="cpu",
            )
            logger.info("reviewer device: cpu (fallback)")
            return pipe
        raise


def review(description: str) -> dict[str, Any]:
    """
    Run the reviewer agent on a transaction description.
    Returns {top1, top1_score, ranking} where ranking is a list of
    (category, score) tuples sorted by score descending.
    """
    try:
        pipe = _get_pipeline()
        result = pipe(
            description,
            candidate_labels=get_all_categories(),
            hypothesis_template=_HYPOTHESIS_TEMPLATE,
            multi_label=False,
        )
        labels = result["labels"]
        scores = result["scores"]
        ranking = list(zip(labels, scores))
        return {
            "top1": labels[0],
            "top1_score": float(scores[0]),
            "ranking": ranking,
        }
    except Exception as e:
        logger.exception("reviewer failed: %s", e)
        return {
            "top1": None,
            "top1_score": 0.0,
            "ranking": [],
            "error": str(e),
        }
