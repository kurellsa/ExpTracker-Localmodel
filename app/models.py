from sqlalchemy import Column, Integer, String, Float, Date, Boolean, Text, DateTime
from sqlalchemy.sql import func
from app.database import Base


SCHEDULE_C_CATEGORIES = [
    "Car & Truck (Actual)",
    "Depreciation / Section 179",
    "Insurance",
    "Legal & Professional",
    "Office Expense",
    "Rent",
    "Supplies",
    "Travel",
    "Meals (50% deductible)",
    "Utilities",
    "Contract Labor / Wages",
    "Advertising & Marketing",
    "Other Business Expense",
    "PERSONAL (excluded)",
]

MEALS_CATEGORY = "Meals (50% deductible)"
MEALS_DEDUCTIBLE_PCT = 0.50


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False)
    description = Column(String(500), nullable=False)
    amount = Column(Float, nullable=False)          # positive = expense
    bank = Column(String(100), nullable=True)
    account = Column(String(100), nullable=True)
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
