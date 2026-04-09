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
    _migrate_schema()
    seed_categories()


def _migrate_schema():
    """SQLite-safe column additions for existing DBs."""
    with engine.begin() as conn:
        cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(transactions)")}
        # Drop-and-ignore for is_income left over from the previous iteration.
        # SQLite can't DROP COLUMN on older versions; leaving it in place is harmless.
        if "account_type" not in cols:
            conn.exec_driver_sql(
                "ALTER TABLE transactions ADD COLUMN account_type VARCHAR(20) NOT NULL DEFAULT 'debit'"
            )
        if "is_inflow" not in cols:
            conn.exec_driver_sql(
                "ALTER TABLE transactions ADD COLUMN is_inflow BOOLEAN NOT NULL DEFAULT 0"
            )


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
