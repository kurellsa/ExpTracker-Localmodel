import re

from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
import os

from app.database import SessionLocal
from app.models import Transaction
from app.services.csv_parser import parse_csv
from app.services.categorizer import categorize_batch, check_ollama

CC_PAYMENT_CATEGORY = "Credit Card Payment (transfer)"
_CC_PAYMENT_RE = re.compile(
    r"\b(autopay|payment\s+thank\s*you|bill\s+payment|online\s+payment|mobile\s+payment|electronic\s+payment|payment\s*-\s*thank)\b",
    re.IGNORECASE,
)


def _is_cc_payment(description: str) -> bool:
    return bool(_CC_PAYMENT_RE.search(description or ""))

router = APIRouter()
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@router.get("/upload")
def upload_page(request: Request):
    ollama_ok = check_ollama()
    return templates.TemplateResponse("upload.html", {"request": request, "ollama_ok": ollama_ok})


@router.post("/upload")
async def handle_upload(
    request: Request,
    files: list[UploadFile] = File(...),
    tax_year: int = Form(2025),
    account_label: str = Form(""),
    account_type: str = Form("debit"),
):
    account_type = "credit" if account_type == "credit" else "debit"
    db = SessionLocal()
    stats = {"imported": 0, "skipped_duplicate": 0, "errors": 0, "files": []}

    try:
        for file in files:
            content = await file.read()
            try:
                rows = parse_csv(content, file.filename)
            except Exception as e:
                stats["errors"] += 1
                stats["files"].append({"name": file.filename, "status": f"Parse error: {e}"})
                continue

            # Dedup first, then batch-categorize only the new rows
            new_rows = []
            dupes = 0
            for row in rows:
                exists = (
                    db.query(Transaction)
                    .filter(
                        Transaction.date == row["date"],
                        Transaction.description == row["description"],
                        Transaction.amount == row["amount"],
                        Transaction.bank == row["bank"],
                        Transaction.tax_year == tax_year,
                    )
                    .first()
                )
                if exists:
                    dupes += 1
                else:
                    new_rows.append(row)

            # For credit-card accounts, pre-route card payments (autopay, etc.)
            # to a dedicated transfer category so they don't hit Schedule C.
            rows_to_categorize = []
            row_precats = []  # parallel: pre-assigned category or None
            for row in new_rows:
                if account_type == "credit" and _is_cc_payment(row["description"]):
                    row_precats.append(CC_PAYMENT_CATEGORY)
                else:
                    row_precats.append(None)
                    rows_to_categorize.append(row)

            categorized_iter = iter(categorize_batch(rows_to_categorize) if rows_to_categorize else [])

            imported = 0
            for row, precat in zip(new_rows, row_precats):
                if precat is not None:
                    cat_result = {
                        "category": precat,
                        "is_approved": True,
                        "llm_category": precat,
                        "llm_confidence": "high",
                        "llm_reasoning": "Auto-flagged as credit-card payment (transfer).",
                    }
                else:
                    cat_result = next(categorized_iter)

                txn = Transaction(
                    date=row["date"],
                    description=row["description"],
                    amount=row["amount"],
                    is_inflow=bool(row.get("is_inflow", False)),
                    bank=row["bank"],
                    account=account_label or row.get("account", ""),
                    account_type=account_type,
                    tax_year=tax_year,
                    category=cat_result["category"],
                    is_personal=(cat_result["category"] == "PERSONAL (excluded)"),
                    is_approved=cat_result.get("is_approved", False),
                    llm_category=cat_result.get("llm_category", cat_result["category"]),
                    llm_confidence=cat_result.get("llm_confidence", cat_result.get("confidence", "low")),
                    llm_reasoning=cat_result.get("llm_reasoning", cat_result.get("reasoning", "")),
                )
                db.add(txn)
                imported += 1

            db.commit()
            stats["imported"] += imported
            stats["skipped_duplicate"] += dupes
            stats["files"].append({
                "name": file.filename,
                "status": f"{imported} imported, {dupes} duplicates skipped",
            })
    finally:
        db.close()

    return templates.TemplateResponse(
        "upload.html",
        {"request": request, "ollama_ok": True, "stats": stats},
    )
