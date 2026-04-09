import csv
import io
from datetime import date
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


def _parse_date(s: str | None):
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _build_schedule_c(
    db,
    year: int,
    account: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict:
    txn_q = db.query(
        Transaction.category, func.sum(Transaction.amount), func.count(Transaction.id)
    ).filter(
        Transaction.tax_year == year,
        Transaction.is_personal == False,  # noqa: E712
        Transaction.category.isnot(None),
        Transaction.category != "PERSONAL (excluded)",
        Transaction.category != "Credit Card Payment (transfer)",
    )
    if account:
        txn_q = txn_q.filter(Transaction.account == account)
    if start_date:
        txn_q = txn_q.filter(Transaction.date >= start_date)
    if end_date:
        txn_q = txn_q.filter(Transaction.date <= end_date)
    rows = txn_q.group_by(Transaction.category).all()

    by_category = {}
    for cat, total, count in rows:
        gross = round(total, 2)
        if cat == MEALS_CATEGORY:
            deductible = round(gross * MEALS_DEDUCTIBLE_PCT, 2)
        else:
            deductible = gross
        by_category[cat] = {"gross": gross, "deductible": deductible, "count": count}

    # Mileage — excluded when filtering by account (mileage isn't account-scoped)
    if account:
        total_miles = 0.0
    else:
        mileage_q = db.query(MileageLog).filter(MileageLog.tax_year == year)
        if start_date:
            mileage_q = mileage_q.filter(MileageLog.date >= start_date)
        if end_date:
            mileage_q = mileage_q.filter(MileageLog.date <= end_date)
        total_miles = sum(r.miles for r in mileage_q.all())
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
def reports_page(
    request: Request,
    year: int = 2025,
    account: str = "",
    start_date: str = "",
    end_date: str = "",
):
    db = SessionLocal()
    try:
        sd = _parse_date(start_date)
        ed = _parse_date(end_date)
        schedule_c = _build_schedule_c(db, year, account or None, sd, ed)

        accounts = [
            row[0] for row in db.query(Transaction.account)
            .filter(Transaction.tax_year == year, Transaction.account.isnot(None), Transaction.account != "")
            .distinct()
            .order_by(Transaction.account)
            .all()
        ]

        pending_q = db.query(func.count(Transaction.id)).filter(
            Transaction.tax_year == year,
            Transaction.is_approved == False,  # noqa: E712
        )
        if account:
            pending_q = pending_q.filter(Transaction.account == account)
        if sd:
            pending_q = pending_q.filter(Transaction.date >= sd)
        if ed:
            pending_q = pending_q.filter(Transaction.date <= ed)
        pending = pending_q.scalar()

        return templates.TemplateResponse(
            "reports.html",
            {
                "request": request,
                "year": year,
                "schedule_c": schedule_c,
                "categories": get_all_categories(db),
                "pending_review": pending,
                "accounts": accounts,
                "filter_account": account,
                "filter_start_date": start_date,
                "filter_end_date": end_date,
            },
        )
    finally:
        db.close()


@router.get("/reports/export")
def export_csv(year: int = 2025, account: str = "", start_date: str = "", end_date: str = ""):
    db = SessionLocal()
    try:
        sd = _parse_date(start_date)
        ed = _parse_date(end_date)
        schedule_c = _build_schedule_c(db, year, account or None, sd, ed)

        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([f"Schedule C Business Expense Summary — Tax Year {year}"])
        writer.writerow([])
        writer.writerow(["IRS Schedule C Category", "Gross Amount", "Deductible Amount", "# Transactions", "Notes"])

        for cat in get_all_categories(db):
            if cat in ("PERSONAL (excluded)", "Credit Card Payment (transfer)"):
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
def export_all_transactions(year: int = 2025, account: str = "", start_date: str = "", end_date: str = ""):
    db = SessionLocal()
    try:
        sd = _parse_date(start_date)
        ed = _parse_date(end_date)
        q = db.query(Transaction).filter(
            Transaction.tax_year == year,
            Transaction.is_personal == False,  # noqa: E712
            Transaction.category != "Credit Card Payment (transfer)",
        )
        if account:
            q = q.filter(Transaction.account == account)
        if sd:
            q = q.filter(Transaction.date >= sd)
        if ed:
            q = q.filter(Transaction.date <= ed)
        txns = q.order_by(Transaction.date).all()

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
