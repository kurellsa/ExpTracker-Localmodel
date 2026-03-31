from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
import os

from app.database import SessionLocal
from app.models import Transaction
from app.services.csv_parser import parse_csv
from app.services.categorizer import categorize_transaction, check_ollama

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
):
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

            imported = 0
            dupes = 0
            for row in rows:
                # Dedup: same date + description + amount + bank
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
                    continue

                # LLM categorize
                cat_result = categorize_transaction(row["description"], row["amount"])

                txn = Transaction(
                    date=row["date"],
                    description=row["description"],
                    amount=row["amount"],
                    bank=row["bank"],
                    account=account_label or row.get("account", ""),
                    tax_year=tax_year,
                    category=cat_result["category"],
                    is_personal=(cat_result["category"] == "PERSONAL (excluded)"),
                    is_approved=False,
                    llm_category=cat_result["category"],
                    llm_confidence=cat_result.get("confidence", "low"),
                    llm_reasoning=cat_result.get("reasoning", ""),
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
