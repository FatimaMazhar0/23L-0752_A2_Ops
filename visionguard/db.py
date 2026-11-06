import datetime as dt
import os

from sqlalchemy import DateTime, String, create_engine, func, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), index=True)
    filename: Mapped[str] = mapped_column(String(120))
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int]
    label: Mapped[str] = mapped_column(String(8))
    p_ai: Mapped[float]
    confidence: Mapped[float]
    threshold: Mapped[float]
    model_name: Mapped[str] = mapped_column(String(80))
    model_version: Mapped[str] = mapped_column(String(40))


_engine = None
_Session = None


def init(url):
    global _engine, _Session
    if url.startswith("sqlite:///"):
        folder = os.path.dirname(url[len("sqlite:///"):])
        if folder:
            os.makedirs(folder, exist_ok=True)
        _engine = create_engine(url, connect_args={"check_same_thread": False})
    else:
        _engine = create_engine(url, pool_pre_ping=True)
    Base.metadata.create_all(_engine)
    _Session = sessionmaker(_engine, expire_on_commit=False)


def add_prediction(**fields):
    with _Session.begin() as s:
        s.add(Prediction(**fields))


def recent(limit=100):
    with _Session() as s:
        rows = s.scalars(select(Prediction).order_by(Prediction.id.desc()).limit(limit)).all()
        total = s.scalar(select(func.count()).select_from(Prediction))
    return rows, total


def ping():
    with _engine.connect() as c:
        c.execute(text("SELECT 1"))