from fastapi import APIRouter, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from datetime import date
import os

from app.database import SessionLocal
from app.models import MileageLog

router = APIRouter()
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

STANDARD_RATES = {2024: 0.67, 2025: 0.70}


@router.get("/mileage")
def mileage_page(request: Request, year: int = 2025):
    db = SessionLocal()
    try:
        logs = db.query(MileageLog).filter(MileageLog.tax_year == year).order_by(MileageLog.date.desc()).all()
        total_miles = sum(r.miles for r in logs)
        rate = STANDARD_RATES.get(year, 0.70)
        total_deduction = round(total_miles * rate, 2)

        return templates.TemplateResponse(
            "mileage.html",
            {
                "request": request,
                "logs": logs,
                "year": year,
                "total_miles": total_miles,
                "total_deduction": total_deduction,
                "rate": rate,
            },
        )
    finally:
        db.close()


@router.post("/mileage/add")
def add_mileage(
    trip_date: str = Form(...),
    from_location: str = Form(...),
    to_location: str = Form(...),
    miles: float = Form(...),
    purpose: str = Form(...),
    year: int = Form(2025),
):
    db = SessionLocal()
    try:
        from datetime import datetime
        parsed_date = datetime.strptime(trip_date, "%Y-%m-%d").date()
        log = MileageLog(
            date=parsed_date,
            from_location=from_location.strip(),
            to_location=to_location.strip(),
            miles=miles,
            purpose=purpose.strip(),
            tax_year=year,
        )
        db.add(log)
        db.commit()
    finally:
        db.close()
    return RedirectResponse(url=f"/mileage?year={year}", status_code=303)


@router.post("/mileage/{log_id}/delete")
def delete_mileage(log_id: int, year: int = Form(2025)):
    db = SessionLocal()
    try:
        log = db.query(MileageLog).get(log_id)
        if log:
            db.delete(log)
            db.commit()
    finally:
        db.close()
    return RedirectResponse(url=f"/mileage?year={year}", status_code=303)
