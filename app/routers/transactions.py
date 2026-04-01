from typing import List

from fastapi import APIRouter, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
import os

from app.database import SessionLocal
from app.models import Transaction, SCHEDULE_C_CATEGORIES

router = APIRouter()
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@router.get("/transactions")
def transactions_page(
    request: Request,
    year: int = 2025,
    category: str = "",
    approved: str = "",
    search: str = "",
    account: str = "",
    page: int = 1,
):
    db = SessionLocal()
    try:
        # Distinct accounts for the filter dropdown
        account_rows = (
            db.query(Transaction.account)
            .filter(Transaction.tax_year == year, Transaction.account != "", Transaction.account.isnot(None))
            .distinct()
            .order_by(Transaction.account)
            .all()
        )
        accounts = [r[0] for r in account_rows]

        q = db.query(Transaction).filter(Transaction.tax_year == year)
        if category:
            q = q.filter(Transaction.category == category)
        if approved == "pending":
            q = q.filter(Transaction.is_approved == False)  # noqa: E712
        elif approved == "approved":
            q = q.filter(Transaction.is_approved == True)  # noqa: E712
        if search:
            q = q.filter(Transaction.description.ilike(f"%{search}%"))
        if account:
            q = q.filter(Transaction.account == account)

        total = q.count()
        per_page = 100
        txns = q.order_by(Transaction.date.desc()).offset((page - 1) * per_page).limit(per_page).all()

        return templates.TemplateResponse(
            "transactions.html",
            {
                "request": request,
                "transactions": txns,
                "categories": SCHEDULE_C_CATEGORIES,
                "accounts": accounts,
                "year": year,
                "filter_category": category,
                "filter_approved": approved,
                "filter_account": account,
                "search": search,
                "page": page,
                "total": total,
                "per_page": per_page,
                "total_pages": max(1, (total + per_page - 1) // per_page),
            },
        )
    finally:
        db.close()


@router.post("/transactions/{txn_id}/approve")
def approve(txn_id: int, year: int = Form(2025)):
    db = SessionLocal()
    try:
        txn = db.query(Transaction).get(txn_id)
        if txn:
            txn.is_approved = True
            db.commit()
    finally:
        db.close()
    return RedirectResponse(url=f"/transactions?year={year}&approved=pending", status_code=303)


@router.post("/transactions/{txn_id}/update")
def update_transaction(
    txn_id: int,
    category: str = Form(...),
    is_personal: str = Form("off"),
    year: int = Form(2025),
):
    db = SessionLocal()
    try:
        txn = db.query(Transaction).get(txn_id)
        if txn:
            txn.category = category
            txn.is_personal = is_personal == "on"
            txn.is_approved = True
            db.commit()
    finally:
        db.close()
    return RedirectResponse(url=f"/transactions?year={year}&approved=pending", status_code=303)


@router.post("/transactions/{txn_id}/delete")
def delete_transaction(txn_id: int, year: int = Form(2025)):
    db = SessionLocal()
    try:
        txn = db.query(Transaction).get(txn_id)
        if txn:
            db.delete(txn)
            db.commit()
    finally:
        db.close()
    return RedirectResponse(url=f"/transactions?year={year}", status_code=303)


@router.post("/transactions/delete-selected")
def delete_selected(
    year: int = Form(2025),
    txn_ids: List[int] = Form(...),
):
    db = SessionLocal()
    try:
        db.query(Transaction).filter(Transaction.id.in_(txn_ids)).delete(
            synchronize_session=False
        )
        db.commit()
    finally:
        db.close()
    return RedirectResponse(url=f"/transactions?year={year}", status_code=303)


@router.post("/transactions/delete-filtered")
def delete_filtered(
    year: int = Form(2025),
    category: str = Form(""),
    approved: str = Form(""),
    search: str = Form(""),
    account: str = Form(""),
):
    db = SessionLocal()
    try:
        q = db.query(Transaction).filter(Transaction.tax_year == year)
        if category:
            q = q.filter(Transaction.category == category)
        if approved == "pending":
            q = q.filter(Transaction.is_approved == False)  # noqa: E712
        elif approved == "approved":
            q = q.filter(Transaction.is_approved == True)  # noqa: E712
        if search:
            q = q.filter(Transaction.description.ilike(f"%{search}%"))
        if account:
            q = q.filter(Transaction.account == account)
        q.delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()
    return RedirectResponse(url=f"/transactions?year={year}", status_code=303)


@router.post("/transactions/approve-all")
def approve_all(year: int = Form(2025)):
    db = SessionLocal()
    try:
        db.query(Transaction).filter(
            Transaction.tax_year == year,
            Transaction.is_approved == False,  # noqa: E712
            Transaction.llm_confidence == "high",
        ).update({"is_approved": True})
        db.commit()
    finally:
        db.close()
    return RedirectResponse(url=f"/transactions?year={year}", status_code=303)
