from sqlalchemy import Column, Integer, String, Float, Date, Boolean, Text, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import Session
from app.database import Base, SessionLocal


SCHEDULE_C_CATEGORIES = [
    # Vehicle / Travel
    "Car & Truck (Actual)",
    "Travel",
    # Capital / Equipment
    "Depreciation / Section 179",
    # Operating expenses
    "Insurance",
    "Legal & Professional",
    "Office Expense",
    "Rent",
    "Supplies",
    "Telephone / Cell Phone",
    "Utilities",
    "Shipping & Freight / Postage",
    "Bank Charges & Fees",
    "Dues & Subscriptions",
    "Interest Expense",
    # People
    "Officer Compensation",
    "Salaries & Wages (non-shareholder)",
    "Contract Labor / Wages",
    "Subcontractors / Outside Services",
    "Payroll Taxes (Employer)",
    "Employee Benefits / Health Insurance",
    "Retirement (SEP / 401k Employer)",
    # Sales / COGS
    "Cost of Goods Sold",
    "Advertising & Marketing",
    # Meals
    "Meals (50% deductible)",
    # Catch-all
    "Other Business Expense",
    "PERSONAL (excluded)",
    "Credit Card Payment (transfer)",
]

MEALS_CATEGORY = "Meals (50% deductible)"
MEALS_DEDUCTIBLE_PCT = 0.50


class Category(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True)
    created_at = Column(DateTime, server_default=func.now())


def get_all_categories(db: Session | None = None) -> list[str]:
    """Return all category names ordered by id (defaults first, then user-added).

    If no session is passed, opens its own. Falls back to the seed constant
    if the table is empty (e.g., tests / before init_db has run).
    """
    own_session = False
    if db is None:
        db = SessionLocal()
        own_session = True
    try:
        rows = db.query(Category.name).order_by(Category.id).all()
        names = [r[0] for r in rows]
        return names if names else list(SCHEDULE_C_CATEGORIES)
    finally:
        if own_session:
            db.close()


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False)
    description = Column(String(500), nullable=False)
    amount = Column(Float, nullable=False)          # always stored positive
    bank = Column(String(100), nullable=True)
    account = Column(String(100), nullable=True)
    account_type = Column(String(20), nullable=False, default="debit")  # "debit" or "credit"
    is_inflow = Column(Boolean, default=False, nullable=False)  # raw direction: True = money came in (refund/deposit)
    tax_year = Column(Integer, nullable=False, default=2025)

    # Categorization
    category = Column(String(100), nullable=True)
    is_personal = Column(Boolean, default=False)
    is_approved = Column(Boolean, default=False)    # user confirmed this category

    # LLM metadata
    llm_category = Column(String(100), nullable=True)   # what LLM suggested
    llm_confidence = Column(String(20), nullable=True)  # high / medium / low
    llm_reasoning = Column(Text, nullable=True)

    created_at = Column(DateTime, server_default=func.now())


class MileageLog(Base):
    __tablename__ = "mileage_log"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False)
    from_location = Column(String(200), nullable=False)
    to_location = Column(String(200), nullable=False)
    miles = Column(Float, nullable=False)
    purpose = Column(String(500), nullable=False)
    tax_year = Column(Integer, nullable=False, default=2025)

    # 2025 IRS standard mileage rate: $0.70/mile (2024 was $0.67)
    STANDARD_RATE_2025 = 0.70
    STANDARD_RATE_2024 = 0.67

    @property
    def deduction_standard(self) -> float:
        rate = self.STANDARD_RATE_2025 if self.tax_year == 2025 else self.STANDARD_RATE_2024
        return round(self.miles * rate, 2)

    created_at = Column(DateTime, server_default=func.now())
