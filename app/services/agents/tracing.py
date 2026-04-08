"""
LangSmith tracing setup.

Privacy: raw merchant descriptions are NEVER sent to LangSmith. Only category
names, confidences, and timings are traced. Inputs containing transaction
descriptions are replaced with a short hash so runs can be correlated without
exposing the original text.
"""
from __future__ import annotations

import hashlib
import logging
import os

logger = logging.getLogger(__name__)


def _hash_description(desc: str) -> str:
    return "desc_" + hashlib.sha256(desc.encode("utf-8")).hexdigest()[:12]


def redact_inputs(inputs: dict) -> dict:
    """Replace raw description with a stable hash before tracing."""
    if not isinstance(inputs, dict):
        return inputs
    redacted = dict(inputs)
    if "description" in redacted and isinstance(redacted["description"], str):
        redacted["description"] = _hash_description(redacted["description"])
    return redacted


def configure_langsmith() -> bool:
    """
    Enable LangSmith tracing if configured in the environment.

    Returns True if tracing is enabled, False otherwise. Safe to call multiple
    times. If LANGSMITH_TRACING is not 'true' or the API key is missing, this
    is a no-op and the graph still runs untraced.
    """
    enabled = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()

    if not enabled or not api_key:
        logger.info("LangSmith tracing disabled")
        return False

    # LangSmith SDK reads these env vars automatically; set the standard ones.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = os.getenv(
        "LANGSMITH_PROJECT", "expense-tracker-categorizer"
    )
    endpoint = os.getenv("LANGSMITH_ENDPOINT", "").strip()
    if endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint

    logger.info(
        "LangSmith tracing enabled (project=%s)",
        os.environ["LANGCHAIN_PROJECT"],
    )
    return True
