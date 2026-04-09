from typing import List

from fastapi import APIRouter, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
import os

from app.database import SessionLocal
from app.models import Transaction, Category, get_all_categories

router = APIRouter()
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


_SORT_COLUMNS = {
    "date": Transaction.date,
    "description": Transaction.description,
    "amount": Transaction.amount,
    "account": Transaction.account,
    "category": Transaction.category,
    "status": Transaction.is_approved,
}


@router.get("/transactions")
def transactions_page(
    request: Request,
    year: int = 2025,
    category: str = "",
    approved: str = "",
    search: str = "",
    account: str = "",
    sort: str = "date",
    dir: str = "desc",
    page: int = 1,
):
    db = SessionLocal()
    try:
        # Get distinct account labels for the filter dropdown
        accounts = [
            row[0] for row in db.query(Transaction.account)
            .filter(Transaction.tax_year == year, Transaction.account.isnot(None), Transaction.account != "")
            .distinct()
            .order_by(Transaction.account)
            .all()
        ]

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

        # Sorting
        sort_key = sort if sort in _SORT_COLUMNS else "date"
        sort_dir = "asc" if dir == "asc" else "desc"
        col = _SORT_COLUMNS[sort_key]
        order = col.asc() if sort_dir == "asc" else col.desc()

        total = q.count()
        per_page = 100
        txns = q.order_by(order, Transaction.id.desc()).offset((page - 1) * per_page).limit(per_page).all()

        return templates.TemplateResponse(
            "transactions.html",
            {
                "request": request,
                "transactions": txns,
                "categories": get_all_categories(db),
                "accounts": accounts,
                "year": year,
                "filter_category": category,
                "filter_approved": approved,
                "filter_account": account,
                "search": search,
                "sort": sort_key,
                "dir": sort_dir,
                "page": page,
                "total": total,
                "per_page": per_page,
                "total_pages": max(1, (total + per_page - 1) // per_page),
            },
        )
    finally:
        db.close()


@router.post("/categories")
def add_category(name: str = Form(...), year: int = Form(2025)):
    """Create a new custom category. Idempotent on case-insensitive duplicates."""
    name = name.strip()[:100]
    if not name:
        return RedirectResponse(url=f"/transactions?year={year}", status_code=303)
    db = SessionLocal()
    try:
        existing = (
            db.query(Category).filter(Category.name.ilike(name)).first()
        )
        if existing is None:
            db.add(Category(name=name))
            db.commit()
    finally:
        db.close()
    return RedirectResponse(url=f"/transactions?year={year}", status_code=303)


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
