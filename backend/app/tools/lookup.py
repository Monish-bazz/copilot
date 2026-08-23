from sqlmodel import Session, select
from app.db.engine import engine
from app.db.models import Account, Order, Ticket
from app.tools.acl import enforce, scoped_account_ids, ForbiddenError
from typing import Optional


def _serialize_row(row) -> dict:
    """Serialize a SQLModel row to dict with ISO datetimes."""
    dump = row.model_dump()
    for k, v in dump.items():
        if hasattr(v, "isoformat"):
            dump[k] = v.isoformat()
    return dump


def get_account(account_id: str, ctx) -> dict:
    try:
        enforce(account_id, ctx)
    except ForbiddenError:
        return {"error": "access_denied", "message": "You do not have access to this account."}

    with Session(engine) as session:
        account = session.get(Account, account_id)
        if not account:
            return {"error": "not_found", "message": "Account not found."}
        return _serialize_row(account)


def get_order(order_id: str, ctx) -> dict:
    with Session(engine) as session:
        order = session.get(Order, order_id)
        if not order:
            return {"error": "not_found", "message": "Order not found."}

        try:
            enforce(order.account_id, ctx)
        except ForbiddenError:
            return {"error": "not_found", "message": "Order not found."}  # no existence leak

        return _serialize_row(order)


def get_ticket(ticket_id: str, ctx) -> dict:
    with Session(engine) as session:
        ticket = session.get(Ticket, ticket_id)
        if not ticket:
            return {"error": "not_found", "message": "Ticket not found."}

        try:
            enforce(ticket.account_id, ctx)
        except ForbiddenError:
            return {"error": "not_found", "message": "Ticket not found."}

        return _serialize_row(ticket)


def query_orders(status: Optional[str] = None, account_id: Optional[str] = None, ctx=None) -> list[dict]:
    allowed = scoped_account_ids(ctx)
    if allowed is not None:
        # Customer: restrict to their own accounts only
        if account_id and account_id not in allowed:
            return [{"error": "access_denied", "message": "You do not have access to this account's orders."}]
        if not account_id:
            account_id = allowed[0]

    with Session(engine) as session:
        query = select(Order)
        if status:
            query = query.where(Order.status == status)
        if account_id:
            query = query.where(Order.account_id == account_id)

        results = session.exec(query).all()
        return [_serialize_row(r) for r in results]


def query_tickets(status: Optional[str] = None, priority: Optional[str] = None, account_id: Optional[str] = None, ctx=None) -> list[dict]:
    allowed = scoped_account_ids(ctx)
    if allowed is not None:
        if account_id and account_id not in allowed:
            return [{"error": "access_denied", "message": "You do not have access to this account's tickets."}]
        if not account_id:
            account_id = allowed[0]

    with Session(engine) as session:
        query = select(Ticket)
        if status:
            query = query.where(Ticket.status == status)
        if priority:
            query = query.where(Ticket.priority == priority)
        if account_id:
            query = query.where(Ticket.account_id == account_id)

        results = session.exec(query).all()
        return [_serialize_row(r) for r in results]
