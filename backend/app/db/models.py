from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime


class Account(SQLModel, table=True):
    account_id: str = Field(primary_key=True)
    name: str
    plan: str
    contract_file: Optional[str] = None
    premium_support: bool = False
    csm: Optional[str] = None
    status: Optional[str] = "active"
    notes: Optional[str] = None


class Order(SQLModel, table=True):
    order_id: str = Field(primary_key=True)
    account_id: str = Field(foreign_key="account.account_id")
    status: str
    booked_at: datetime
    pickup_window_start: Optional[datetime] = None
    pickup_window_end: Optional[datetime] = None
    picked_up_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    cancellation_requested_at: Optional[datetime] = None
    carrier: Optional[str] = None
    carrier_fault: bool = False
    customer_fault: bool = False
    amount_inr: float = 0.0
    notes: Optional[str] = None


class Ticket(SQLModel, table=True):
    ticket_id: str = Field(primary_key=True)
    account_id: str = Field(foreign_key="account.account_id")
    order_id: Optional[str] = None
    priority: str
    status: str
    category: str
    subject: str
    description: str
    channel: Optional[str] = None
    assigned_to: Optional[str] = None
    last_customer_message_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    resolution: Optional[str] = None
    sla_hours: float = 48.0


class Escalation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    draft_id: str
    order_id: Optional[str] = None
    ticket_id: Optional[str] = None
    account_id: Optional[str] = None
    priority: str
    reason: str
    status: str = "draft"  # draft, executed, cancelled
    created_by: str
    created_at: datetime


class FollowupTask(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    draft_id: str
    related_id: Optional[str] = None
    description: str
    status: str = "draft"
    created_by: str
    created_at: datetime


class AuditLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_key: str
    tool_name: str
    arguments_json: str
    result_summary: str
    timestamp: datetime
