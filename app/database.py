from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "expenses.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from app.models import Transaction, MileageLog, Category  # noqa: F401
    Base.metadata.create_all(bind=engine)
    seed_categories()


def seed_categories():
    """Upsert the default category list into the categories table.

    Idempotent — only inserts names that don't already exist. Preserves
    insertion order so the default categories keep their logical grouping
    in the dropdown (sorted by id at read time).
    """
    from app.models import Category, SCHEDULE_C_CATEGORIES

    db = SessionLocal()
    try:
        existing = {row[0] for row in db.query(Category.name).all()}
        for name in SCHEDULE_C_CATEGORIES:
            if name not in existing:
                db.add(Category(name=name))
        db.commit()
    finally:
        db.close()
