import logging

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from app.database import init_db
from app.routers import upload, transactions, mileage, reports

app = FastAPI(title="SmartForce Expense Tracker")

BASE_DIR = os.path.dirname(__file__)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@app.on_event("startup")
def on_startup():
    init_db()


app.include_router(upload.router)
app.include_router(transactions.router)
app.include_router(mileage.router)
app.include_router(reports.router)


@app.get("/")
def root():
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard")
def dashboard(request: Request):
    from sqlalchemy.orm import Session
    from app.database import SessionLocal
    from app.models import Transaction, MileageLog, MEALS_CATEGORY, MEALS_DEDUCTIBLE_PCT
    from sqlalchemy import func

    db: Session = SessionLocal()
    try:
        year = 2025

        rows = (
            db.query(Transaction.category, func.sum(Transaction.amount))
            .filter(
                Transaction.tax_year == year,
                Transaction.is_personal == False,  # noqa: E712
                Transaction.category.isnot(None),
            )
            .group_by(Transaction.category)
            .all()
        )

        category_totals = {}
        for cat, total in rows:
            if cat == MEALS_CATEGORY:
                category_totals[cat] = round(total * MEALS_DEDUCTIBLE_PCT, 2)
            else:
                category_totals[cat] = round(total, 2)

        total_deductible = sum(category_totals.values())

        pending_review = (
            db.query(func.count(Transaction.id))
            .filter(Transaction.tax_year == year, Transaction.is_approved == False)  # noqa: E712
            .scalar()
        )

        total_txns = db.query(func.count(Transaction.id)).filter(Transaction.tax_year == year).scalar()

        mileage_rows = db.query(MileageLog).filter(MileageLog.tax_year == year).all()
        total_miles = sum(r.miles for r in mileage_rows)
        mileage_deduction = round(total_miles * 0.70, 2)

        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "year": year,
                "category_totals": category_totals,
                "total_deductible": total_deductible,
                "pending_review": pending_review,
                "total_txns": total_txns,
                "total_miles": total_miles,
                "mileage_deduction": mileage_deduction,
            },
        )
    finally:
        db.close()
