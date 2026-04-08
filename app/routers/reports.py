import csv
import io
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func
import os

from app.database import SessionLocal
from app.models import Transaction, MileageLog, get_all_categories, MEALS_CATEGORY, MEALS_DEDUCTIBLE_PCT

router = APIRouter()
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

STANDARD_RATES = {2024: 0.67, 2025: 0.70}


def _build_schedule_c(db, year: int) -> dict:
    rows = (
        db.query(Transaction.category, func.sum(Transaction.amount), func.count(Transaction.id))
        .filter(
            Transaction.tax_year == year,
            Transaction.is_personal == False,  # noqa: E712
            Transaction.category.isnot(None),
            Transaction.category != "PERSONAL (excluded)",
        )
        .group_by(Transaction.category)
        .all()
    )

    by_category = {}
    for cat, total, count in rows:
        gross = round(total, 2)
        if cat == MEALS_CATEGORY:
            deductible = round(gross * MEALS_DEDUCTIBLE_PCT, 2)
        else:
            deductible = gross
        by_category[cat] = {"gross": gross, "deductible": deductible, "count": count}

    # Mileage
    mileage_rows = db.query(MileageLog).filter(MileageLog.tax_year == year).all()
    total_miles = sum(r.miles for r in mileage_rows)
    rate = STANDARD_RATES.get(year, 0.70)
    mileage_deduction = round(total_miles * rate, 2)

    total_deductible = sum(v["deductible"] for v in by_category.values()) + mileage_deduction

    return {
        "by_category": by_category,
        "total_miles": total_miles,
        "mileage_rate": rate,
        "mileage_deduction": mileage_deduction,
        "total_deductible": total_deductible,
    }


@router.get("/reports")
def reports_page(request: Request, year: int = 2025):
    db = SessionLocal()
    try:
        schedule_c = _build_schedule_c(db, year)

        pending = (
            db.query(func.count(Transaction.id))
            .filter(Transaction.tax_year == year, Transaction.is_approved == False)  # noqa: E712
            .scalar()
        )

        return templates.TemplateResponse(
            "reports.html",
            {
                "request": request,
                "year": year,
                "schedule_c": schedule_c,
                "categories": get_all_categories(db),
                "pending_review": pending,
            },
        )
    finally:
        db.close()


@router.get("/reports/export")
def export_csv(year: int = 2025):
    db = SessionLocal()
    try:
        schedule_c = _build_schedule_c(db, year)

        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([f"Schedule C Business Expense Summary — Tax Year {year}"])
        writer.writerow([])
        writer.writerow(["IRS Schedule C Category", "Gross Amount", "Deductible Amount", "# Transactions", "Notes"])

        for cat in get_all_categories(db):
            if cat == "PERSONAL (excluded)":
                continue
            data = schedule_c["by_category"].get(cat, {"gross": 0, "deductible": 0, "count": 0})
            note = "50% cap applied" if cat == MEALS_CATEGORY else ""
            writer.writerow([cat, f"${data['gross']:.2f}", f"${data['deductible']:.2f}", data["count"], note])

        writer.writerow([])
        writer.writerow([
            "Mileage (Standard Rate)",
            f"{schedule_c['total_miles']:.1f} miles @ ${schedule_c['mileage_rate']}/mi",
            f"${schedule_c['mileage_deduction']:.2f}",
            "",
            "Schedule C Line 9",
        ])
        writer.writerow([])
        writer.writerow(["TOTAL DEDUCTIBLE EXPENSES", "", f"${schedule_c['total_deductible']:.2f}", "", ""])

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=schedule_c_{year}.csv"},
        )
    finally:
        db.close()


@router.get("/reports/export/transactions")
def export_all_transactions(year: int = 2025):
    db = SessionLocal()
    try:
        txns = (
            db.query(Transaction)
            .filter(Transaction.tax_year == year, Transaction.is_personal == False)  # noqa: E712
            .order_by(Transaction.date)
            .all()
        )

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Date", "Description", "Amount", "Category", "Deductible Amount", "Bank", "Account", "Approved", "LLM Confidence"])

        for t in txns:
            deductible = round(t.amount * MEALS_DEDUCTIBLE_PCT, 2) if t.category == MEALS_CATEGORY else t.amount
            writer.writerow([
                t.date, t.description, f"${t.amount:.2f}", t.category,
                f"${deductible:.2f}", t.bank, t.account,
                "Yes" if t.is_approved else "No", t.llm_confidence,
            ])

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=transactions_{year}.csv"},
        )
    finally:
        db.close()
