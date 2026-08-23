import os
import pandas as pd
from sqlmodel import Session
from datetime import datetime
from sqlalchemy import text
from app.config import RAW_DIR
from app.db.engine import engine, init_db
from app.db.models import Account, Order, Ticket


def _parse_dt(val):
    """Safely parse a datetime value from pandas."""
    if pd.isna(val) or val == "" or val is None:
        return None
    try:
        return pd.to_datetime(val)
    except Exception:
        return None


def _parse_bool(val) -> bool:
    """Safely parse a boolean from pandas (handles string 'True'/'False' after fillna)."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    return False


def ingest():
    init_db()
    file_path = os.path.join(RAW_DIR, "ParcelPilot_Assessment_Data.xlsx")

    print("Reading Excel sheets...")
    accounts_df = pd.read_excel(file_path, sheet_name="accounts")
    orders_df = pd.read_excel(file_path, sheet_name="orders")
    tickets_df = pd.read_excel(file_path, sheet_name="tickets")

    with Session(engine) as session:
        print("Clearing old data...")
        session.exec(text("DROP TABLE IF EXISTS escalation CASCADE"))
        session.exec(text("DROP TABLE IF EXISTS followuptask CASCADE"))
        session.exec(text("DROP TABLE IF EXISTS auditlog CASCADE"))
        session.exec(text("DROP TABLE IF EXISTS ticket CASCADE"))
        session.exec(text("DROP TABLE IF EXISTS \"order\" CASCADE"))
        session.exec(text("DROP TABLE IF EXISTS account CASCADE"))
        session.commit()

    init_db()

    with Session(engine) as session:
        # --- Accounts ---
        print("Ingesting accounts...")
        for _, row in accounts_df.iterrows():
            contract = row.get("contract_file")
            if pd.isna(contract) or contract == "":
                contract = None
            else:
                contract = str(contract).strip()

            notes = row.get("notes")
            if pd.isna(notes) or notes == "":
                notes = None
            else:
                notes = str(notes).strip()

            account = Account(
                account_id=str(row["account_id"]).strip(),
                name=str(row.get("account_name", "")).strip(),
                plan=str(row.get("plan", "")).strip(),
                contract_file=contract,
                premium_support=_parse_bool(row.get("premium_support", False)),
                csm=str(row.get("csm", "")).strip() if pd.notna(row.get("csm")) else None,
                status=str(row.get("status", "active")).strip() if pd.notna(row.get("status")) else "active",
                notes=notes,
            )
            session.add(account)

        # --- Orders ---
        print("Ingesting orders...")
        for _, row in orders_df.iterrows():
            booked = _parse_dt(row.get("booked_at"))
            if booked is None:
                booked = datetime(2026, 8, 16, 9, 0)  # fallback

            order = Order(
                order_id=str(row["order_id"]).strip(),
                account_id=str(row["account_id"]).strip(),
                status=str(row["status"]).strip(),
                booked_at=booked,
                pickup_window_start=_parse_dt(row.get("pickup_window_start")),
                pickup_window_end=_parse_dt(row.get("pickup_window_end")),
                picked_up_at=_parse_dt(row.get("pickup_actual_at")),
                delivered_at=_parse_dt(row.get("delivered_at")),
                cancellation_requested_at=_parse_dt(row.get("cancellation_requested_at")),
                carrier=str(row["carrier"]).strip() if pd.notna(row.get("carrier")) else None,
                carrier_fault=_parse_bool(row.get("carrier_fault", False)),
                customer_fault=_parse_bool(row.get("customer_fault", False)),
                amount_inr=float(row.get("shipment_fee_inr", 0.0)) if pd.notna(row.get("shipment_fee_inr")) else 0.0,
                notes=str(row["notes"]).strip() if pd.notna(row.get("notes")) else None,
            )
            session.add(order)

        # --- Tickets ---
        print("Ingesting tickets...")
        for _, row in tickets_df.iterrows():
            created = _parse_dt(row.get("created_at"))
            if created is None:
                created = datetime(2026, 8, 16, 9, 0)

            subject = str(row.get("subject", "")).strip() if pd.notna(row.get("subject")) else ""
            description = str(row.get("description", "")).strip() if pd.notna(row.get("description")) else ""

            # Priority from the spreadsheet (if column exists), else derive
            priority_raw = row.get("priority")
            if pd.notna(priority_raw) and str(priority_raw).strip():
                priority = str(priority_raw).strip().lower()
            else:
                combined = (subject + " " + description).lower()
                if any(k in combined for k in ["outage", "500", "p0", "critical", "security", "api key"]):
                    priority = "p0"
                elif any(k in combined for k in ["high", "urgent"]):
                    priority = "p1"
                else:
                    priority = "p2"

            # Status
            status_raw = row.get("status", "open")
            status = str(status_raw).strip() if pd.notna(status_raw) else "open"

            # Resolution
            resolution = None
            res_val = row.get("historical_resolution")
            if pd.notna(res_val) and str(res_val).strip():
                resolution = str(res_val).strip()

            # Category
            category_raw = row.get("category")
            category = str(category_raw).strip() if pd.notna(category_raw) and str(category_raw).strip() else "general"

            # SLA hours: use column if present, else defaults by priority
            sla_raw = row.get("sla_hours")
            if pd.notna(sla_raw):
                sla_hours = float(sla_raw)
            else:
                sla_defaults = {"p0": 0.5, "p1": 2.0, "p2": 24.0, "p3": 48.0}
                sla_hours = sla_defaults.get(priority, 48.0)

            # Channel and assigned_to
            channel = str(row.get("channel", "")).strip() if pd.notna(row.get("channel")) else None
            assigned_to = str(row.get("assigned_to", "")).strip() if pd.notna(row.get("assigned_to")) else None
            last_msg = _parse_dt(row.get("last_customer_message_at"))

            ticket = Ticket(
                ticket_id=str(row["ticket_id"]).strip(),
                account_id=str(row["account_id"]).strip(),
                order_id=str(row["order_id"]).strip() if pd.notna(row.get("order_id")) and str(row.get("order_id")).strip() else None,
                priority=priority,
                status=status,
                category=category,
                subject=subject,
                description=description,
                channel=channel,
                assigned_to=assigned_to,
                last_customer_message_at=last_msg,
                created_at=created,
                updated_at=created,
                resolution=resolution,
                sla_hours=sla_hours,
            )
            session.add(ticket)

        session.commit()
        print("Data ingested successfully.")


if __name__ == "__main__":
    ingest()
