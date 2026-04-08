# Multi-Agent Transaction Categorizer — Architecture

**Status:** Implemented 2026-04-08
**Scope:** `ExpenseTracker/app/services/agents/` + integration points in `categorizer.py` and `upload.py`

---

## 1. Goals

Categorize imported bank transactions into one of 14 IRS Schedule C categories with:

1. **High accuracy** — tax data needs to be trustworthy; miscategorized rows waste review time.
2. **Low latency** — CSV imports can contain hundreds of rows; categorization shouldn't dominate import time.
3. **Fully local / air-gapped** — raw merchant descriptions must never leave the machine (privacy requirement).
4. **Observable** — when things go wrong, we need to see which agent made the call and why.

## 2. Why Multi-Agent?

The previous implementation used a single 9B generative model (`qwen3.5:9b` via Ollama). Two problems:

- **Wrong tool for the job.** A 9B decoder-only model is overkill for 14-way classification on short text. Bigger ≠ better for narrow classification tasks.
- **High latency.** 9B at Q4 on Apple Silicon is slow; a 500-row CSV import stalls.

Multi-agent with **error decorrelation** fixes both. The key insight: if both agents share the same architecture and training data, their errors will be correlated and a "review" step adds almost no value. We picked two agents from fundamentally different model families so their mistakes are uncorrelated and the reviewer provides a genuine second opinion.

| Property | Classifier (Agent 1) | Reviewer (Agent 2) |
|---|---|---|
| Model | `llama3.2:3b` (Q4_K_M GGUF) | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` |
| Architecture | Decoder-only (causal LM) | Encoder-only (DeBERTa-v2) |
| Training objective | Next-token prediction + instruction tuning | NLI + zero-shot classification |
| Parameters | ~3.2 B | 184 M |
| Runtime | Ollama (llama.cpp + Metal) | HF transformers + PyTorch MPS |
| Output | Free-text JSON w/ reasoning | Ranked scores over all 14 labels |
| Strength | Flexible, handles weird merchant names via reasoning | Purpose-built for "does this text belong to category X" |

## 3. Architecture

```
Transaction (description, amount)
        │
        ▼
┌────────────────────────────────────┐
│ Node: classify                     │
│ Ollama llama3.2:3b                 │
│ → {category, confidence, reasoning}│
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ Node: review                       │
│ DeBERTa-v3-base zeroshot (on MPS)  │
│ → ranked list over 14 Sch.C cats   │
│   + top-1 + top-1 score            │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ Node: resolve                      │
│                                    │
│ agreed = llama_cat == deberta_top1 │
│                                    │
│ if reviewer_failed:                │
│   keep llama, is_approved=False    │
│ elif agreed and llama=="high":     │
│   auto-approve (is_approved=True)  │
│ elif agreed:                       │
│   keep llama, is_approved=False    │
│ else:                              │
│   keep llama, is_approved=False,   │
│   append reviewer note to reason   │
└────────────┬───────────────────────┘
             │
             ▼
   Persisted Transaction row
```

All three nodes live in a single LangGraph `StateGraph` with linear edges `START → classify → review → resolve → END`. No conditional branching — the reviewer runs on every transaction because the user prioritized accuracy over latency. (Conditional branching based on Llama's confidence was considered and rejected.)

## 4. Components

### 4.1 `app/services/agents/classifier_agent.py`
- Thin wrapper over `ollama.chat()`
- Loads model name from `CATEGORIZER_MODEL` env var (default `llama3.2:3b`)
- Temperature 0.1 for deterministic JSON output
- Validates model's category against `SCHEDULE_C_CATEGORIES`; falls back to `"Other Business Expense"` / `"low"` on invalid label or parse error
- Strips markdown fences from model output (defensive — small models sometimes wrap JSON)
- Public: `classify(description, amount) -> {category, confidence, reasoning}`

### 4.2 `app/services/agents/reviewer_agent.py`
- Loads DeBERTa via `transformers.pipeline("zero-shot-classification", ...)`
- **Device selection**: `_select_device()` tries `REVIEWER_DEVICE` env override first, then auto-detects `mps → cuda → cpu`. On Apple Silicon this uses Metal.
- **Fallback**: if the model fails to load on MPS (rare op compatibility issue), automatically retries on CPU with a warning log.
- **Singleton loading**: `@lru_cache(maxsize=1)` on `_get_pipeline()` — weights load once at first call, not per transaction.
- **Hypothesis template**: `"This business expense is for {}."` — phrases each category as a natural-language hypothesis for the NLI model to score.
- Candidate labels are the 14 `SCHEDULE_C_CATEGORIES` from `models.py`.
- Returns top-1 + full ranked list with scores.
- Public: `review(description) -> {top1, top1_score, ranking}`

### 4.3 `app/services/agents/graph.py`
- `CategorizationState` TypedDict carries all intermediate fields
- Three node functions: `_classify_node`, `_review_node`, `_resolve_node`
- Compiled graph exposed as `categorization_graph`
- Public helper: `run_categorization(description, amount)` — invokes the graph and returns a flat dict mapped to `Transaction` model fields (`category`, `confidence`, `reasoning`, `is_approved`, `llm_category`, `llm_confidence`, `llm_reasoning`)
- Calls `configure_langsmith()` at module import time so tracing is live as soon as the graph is used

### 4.4 `app/services/agents/tracing.py`
- `configure_langsmith()` — reads env vars, sets the standard `LANGCHAIN_*` vars that the LangSmith SDK picks up automatically. No-op if `LANGSMITH_TRACING != "true"` or `LANGSMITH_API_KEY` is empty.
- `redact_inputs(inputs)` — replaces raw `description` with `"desc_" + sha256(desc)[:12]` before any serialization. Used to build a redacted `run_name` for each graph invocation.
- **Privacy invariant**: raw merchant descriptions must not appear in any LangSmith trace. Only category names, confidence scores, timings, and the SHA-256-hashed description ID are traceable.

### 4.5 `app/services/categorizer.py` (wrapper)
- Preserves the original public API so `upload.py` and any other caller doesn't need to change:
  - `categorize_transaction(description, amount)` → delegates to `run_categorization`
  - `categorize_batch(transactions)` → iterates over `categorize_transaction`
  - `check_ollama()` → unchanged, used for startup health check
- Kept as a thin wrapper rather than deleted so the graph is an implementation detail callers don't need to know about.

## 5. State Schema

```python
class CategorizationState(TypedDict, total=False):
    # Input
    description: str
    amount: float

    # Classifier (Agent 1) outputs
    llama_category: str
    llama_confidence: str       # "high" | "medium" | "low"
    llama_reasoning: str

    # Reviewer (Agent 2) outputs
    deberta_top1: Optional[str]
    deberta_top1_score: float
    deberta_ranking: list       # [(label, score), ...]

    # Resolver outputs
    agreed: bool
    final_category: str
    final_confidence: str
    final_reasoning: str
    is_approved: bool
```

## 6. Conflict Resolution Policy

Decided by the user during planning:

| Case | Action | `is_approved` |
|---|---|---|
| Reviewer failed (model load, inference exception) | Keep Llama's pick | `False` (flag for manual review) |
| Agents agree **and** Llama confidence is `high` | Keep Llama's pick | `True` (auto-approve) |
| Agents agree but Llama confidence is `medium` or `low` | Keep Llama's pick | `False` |
| Agents disagree | Keep Llama's pick, append reviewer note to `llm_reasoning` | `False` |

Llama always wins on category — the reviewer's role is to flag doubt, not to override. User's rationale: Llama can reason about weird merchant strings; DeBERTa only scores similarity.

## 7. Integration Points

### Upload flow (`app/routers/upload.py`)
```python
cat_result = categorize_transaction(row["description"], row["amount"])

txn = Transaction(
    ...
    category=cat_result["category"],
    is_personal=(cat_result["category"] == "PERSONAL (excluded)"),
    is_approved=cat_result.get("is_approved", False),
    llm_category=cat_result.get("llm_category", cat_result["category"]),
    llm_confidence=cat_result.get("llm_confidence", cat_result.get("confidence", "low")),
    llm_reasoning=cat_result.get("llm_reasoning", cat_result.get("reasoning", "")),
)
```

No database schema migration — the existing `Transaction` fields (`category`, `llm_category`, `llm_confidence`, `llm_reasoning`, `is_approved`) cover everything the graph produces. The reviewer's verdict is written into `llm_reasoning` alongside Llama's original reasoning.

### Transactions UI (`app/routers/transactions.py` + template)
Unchanged. The UI already reads `llm_reasoning` as a hover tooltip on the description column, so reviewer disagreements become visible there for free. Rows with `is_approved=False` are surfaced via the existing "Needs Review" filter.

## 8. Configuration (`.env`)

```bash
# Models
CATEGORIZER_MODEL=llama3.2:3b
REVIEWER_MODEL=MoritzLaurer/deberta-v3-base-zeroshot-v2.0
REVIEWER_DEVICE=            # empty = auto (mps → cuda → cpu)

# LangSmith (disabled by default)
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=expense-tracker-categorizer
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

`.env` is gitignored. Never commit API keys.

## 9. Performance Characteristics

Rough numbers on Apple Silicon (actual will vary):

| Stage | Baseline (`qwen3.5:9b`) | New pipeline |
|---|---|---|
| Classifier | ~2-4 s / txn | ~0.5-1 s / txn (Llama 3.2:3b) |
| Reviewer | n/a | ~15-30 ms / txn (DeBERTa on MPS) |
| **Total per txn** | **~2-4 s** | **~0.5-1 s** |

The reviewer runs on every transaction but adds only tens of milliseconds because DeBERTa-base is small and runs on MPS. The classifier swap (9B → 3B) is where the real latency win comes from.

## 10. Privacy & Security

- **All inference is local.** Ollama runs llama.cpp locally, transformers runs in-process. No cloud LLM calls.
- **LangSmith tracing is opt-in** (disabled by default in `.env`).
- **When tracing is enabled**, the `redact_inputs()` function replaces raw `description` with a SHA-256 hash prefix. The run name sent to LangSmith looks like `categorize/desc_a1b2c3d4e5f6` — correlatable across runs (same merchant → same hash) but the original text is never exposed.
- **No secrets in code.** All keys live in `.env`, which is gitignored.

## 11. Failure Modes & Handling

| Failure | Behavior |
|---|---|
| Ollama not running / model missing | Classifier returns `{"category": "Other Business Expense", "confidence": "low", "reasoning": "Classifier error: ..."}` — graph continues, reviewer still runs, resolve step flags for manual review |
| DeBERTa weights fail to download (first run, no network) | Reviewer returns `{top1: None, ...}` — resolve step keeps Llama's pick and sets `is_approved=False` |
| MPS op not supported in current torch build | Reviewer's `_get_pipeline()` catches, logs warning, retries on CPU |
| LangSmith API unreachable | Tracing SDK fails silently; graph still runs normally |
| Model returns malformed JSON | Classifier's `_parse_response` catches JSON decode error, returns fallback |

## 12. Setup

```bash
cd ExpenseTracker
ollama pull llama3.2:3b           # ~2 GB download
pip install -r requirements.txt   # installs langgraph, langsmith, transformers, torch
```

First run will auto-download DeBERTa weights (~370 MB) to `~/.cache/huggingface/`.

Optional — enable tracing:
```bash
# in .env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ls__...
```

No code change needed; `tracing.py` reads env vars at import time.

## 13. Verification Checklist

- [ ] `ollama run llama3.2:3b "hello"` returns a response
- [ ] Scratch test: `python -c "from app.services.categorizer import categorize_transaction; print(categorize_transaction('STARBUCKS STORE #4521', 6.75))"` returns a dict with `category`, `is_approved`, `llm_reasoning`
- [ ] Startup logs show `reviewer device: mps` (not `cpu`)
- [ ] Upload a 5-10 row CSV, verify the Transactions UI shows categories, confidence badges, and hover-tooltip reasoning
- [ ] At least one disagreement case visible in the UI (shown as `is_approved=False` with reviewer note in tooltip)
- [ ] With tracing enabled, upload a CSV containing `ZZZ_TEST_MERCHANT_9999` and search LangSmith for that string — it should NOT appear anywhere (privacy audit)

## 14. Future Work (not yet implemented)

- **Caching.** Identical `(description, amount)` pairs are classified repeatedly on re-imports. A memoization cache keyed on `(hash(description), amount)` would help.
- **Batch inference.** DeBERTa supports batching; a batched reviewer call could process an entire CSV in one forward pass for further speedup.
- **Confidence-gated reviewer.** Currently runs on every transaction. A LangGraph conditional edge based on Llama confidence would skip the reviewer for high-confidence cases and save ~15-30 ms/txn × N rows.
- **Model fine-tuning.** DeBERTa zero-shot is good but not optimal; fine-tuning on labeled transaction data would likely beat both agents.
