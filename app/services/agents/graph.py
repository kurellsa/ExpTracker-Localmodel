"""
LangGraph orchestration for the multi-agent categorizer.

Flow:  START → classify → review → resolve → END

Every transaction goes through all three nodes. On disagreement between
classifier and reviewer, Llama's pick is kept but is_approved is forced to
False so the UI flags it for manual review; the reviewer's suggestion is
appended to llm_reasoning for visibility.
"""
from __future__ import annotations

from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from app.services.agents import classifier_agent, reviewer_agent
from app.services.agents.tracing import configure_langsmith, redact_inputs

# Configure tracing at import time (no-op if not enabled)
configure_langsmith()


class CategorizationState(TypedDict, total=False):
    # Input
    description: str
    amount: float

    # Classifier outputs
    llama_category: str
    llama_confidence: str
    llama_reasoning: str

    # Reviewer outputs
    deberta_top1: Optional[str]
    deberta_top1_score: float
    deberta_ranking: list

    # Resolver outputs
    agreed: bool
    final_category: str
    final_confidence: str
    final_reasoning: str
    is_approved: bool


def _classify_node(state: CategorizationState) -> dict[str, Any]:
    result = classifier_agent.classify(state["description"], state["amount"])
    return {
        "llama_category": result["category"],
        "llama_confidence": result.get("confidence", "low"),
        "llama_reasoning": result.get("reasoning", ""),
    }


def _review_node(state: CategorizationState) -> dict[str, Any]:
    result = reviewer_agent.review(state["description"])
    return {
        "deberta_top1": result.get("top1"),
        "deberta_top1_score": result.get("top1_score", 0.0),
        "deberta_ranking": result.get("ranking", []),
    }


def _resolve_node(state: CategorizationState) -> dict[str, Any]:
    llama_cat = state["llama_category"]
    llama_conf = state.get("llama_confidence", "low")
    llama_reason = state.get("llama_reasoning", "")
    deberta_top1 = state.get("deberta_top1")
    deberta_score = state.get("deberta_top1_score", 0.0)

    agreed = deberta_top1 is not None and llama_cat == deberta_top1
    final_reasoning = llama_reason

    if deberta_top1 is None:
        # Reviewer failed — keep Llama's pick, don't auto-approve
        is_approved = False
        final_reasoning = (
            f"{llama_reason} | Reviewer unavailable; manual review recommended."
        ).strip(" |")
    elif agreed:
        # Both agree — auto-approve only if Llama was high confidence
        is_approved = llama_conf == "high"
        final_reasoning = (
            f"{llama_reason} | Reviewer agreed (score={deberta_score:.2f})."
        ).strip(" |")
    else:
        # Disagreement — keep Llama's pick per user policy, but flag for review
        is_approved = False
        final_reasoning = (
            f"{llama_reason} | Reviewer disagreed: suggested "
            f"'{deberta_top1}' (score={deberta_score:.2f})."
        ).strip(" |")

    return {
        "agreed": agreed,
        "final_category": llama_cat,
        "final_confidence": llama_conf,
        "final_reasoning": final_reasoning,
        "is_approved": is_approved,
    }


def _build_graph():
    builder = StateGraph(CategorizationState)
    builder.add_node("classify", _classify_node)
    builder.add_node("review", _review_node)
    builder.add_node("resolve", _resolve_node)
    builder.add_edge(START, "classify")
    builder.add_edge("classify", "review")
    builder.add_edge("review", "resolve")
    builder.add_edge("resolve", END)
    return builder.compile()


categorization_graph = _build_graph()


def run_categorization(description: str, amount: float) -> dict[str, Any]:
    """
    Invoke the multi-agent graph and return a flat dict mapped to the
    Transaction model fields used downstream.

    Returned keys:
      category, confidence, reasoning, is_approved,
      llm_category, llm_confidence, llm_reasoning
    """
    initial_state: CategorizationState = {
        "description": description,
        "amount": amount,
    }
    # Use a redacted run name so LangSmith traces don't leak merchant strings.
    run_name = f"categorize/{redact_inputs({'description': description})['description']}"
    final_state = categorization_graph.invoke(
        initial_state,
        config={"run_name": run_name},
    )

    return {
        # Primary fields
        "category": final_state["final_category"],
        "confidence": final_state["final_confidence"],
        "reasoning": final_state["final_reasoning"],
        "is_approved": final_state["is_approved"],
        # LLM metadata (preserves existing Transaction schema)
        "llm_category": final_state["llama_category"],
        "llm_confidence": final_state["llama_confidence"],
        "llm_reasoning": final_state["final_reasoning"],
    }
