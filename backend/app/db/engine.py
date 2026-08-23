from sqlmodel import create_engine, SQLModel
from app.config import DATABASE_URL
import os

_url = DATABASE_URL
if _url and _url.startswith("postgres://"):
    _url = _url.replace("postgres://", "postgresql://", 1)

engine = create_engine(_url, echo=False) if _url else None


def init_db():
    if engine:
        from app.db.models import Account, Order, Ticket, Escalation, FollowupTask, AuditLog
        SQLModel.metadata.create_all(engine)
