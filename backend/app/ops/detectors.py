"""
Proactive issue detectors for the /ops dashboard.
Each detector is a simple rule-based scan over tickets/orders.
"""
from sqlmodel import Session, select
from app.db.engine import engine
from app.db.models import Ticket, Order, Account
from app.config import SNAPSHOT_TS

SNAPSHOT_NAIVE = SNAPSHOT_TS.replace(tzinfo=None)

# SLA targets in MINUTES (not hours) for precision
# These are DEFAULT plan-level SLAs. Custom contracts override.
PLAN_SLA_MINUTES = {
    "Enterprise": {"p0": 15, "p1": 30, "p2": 120, "p3": 1440},
    "Growth": {"p0": 30, "p1": 120, "p2": 240, "p3": 2880},
    "Standard": {"p0": 60, "p1": 240, "p2": 1440, "p3": 2880},
}

# Custom SLA overrides per account (from contracts)
ACCOUNT_SLA_OVERRIDES = {
    "ACCT-001": {"p0": 15, "p1": 15, "p2": 60, "p3": 480},  # Northstar: P1 = 15 min
}


def get_sla_minutes(account_id: str, plan: str, priority: str) -> float:
    """
    Get SLA target in minutes for an account/priority combo.

    Single source of truth: both the /ops dashboard and the chat agent call this,
    so the dashboard and a chat answer can never disagree about a target.
    Contract-derived overrides (extracted from the agreement PDFs at query time)
    should be passed to the calculator directly and take precedence over this.
    """
    priority_key = (priority or "").lower()
    if priority_key in ("critical",):
        priority_key = "p0"

    if account_id in ACCOUNT_SLA_OVERRIDES:
        return ACCOUNT_SLA_OVERRIDES[account_id].get(priority_key, 2880)

    return PLAN_SLA_MINUTES.get(plan, PLAN_SLA_MINUTES["Standard"]).get(priority_key, 2880)


# Backwards-compatible alias
_get_sla_minutes = get_sla_minutes


def run_detectors(ctx) -> list[dict]:
    issues = []

    with Session(engine) as session:
        tickets = session.exec(select(Ticket)).all()
        orders = session.exec(select(Order)).all()
        accounts = session.exec(select(Account)).all()
        acct_map = {a.account_id: a for a in accounts}

        # 1. P0/Critical Outage
        p0_tickets = [t for t in tickets if t.priority.lower() in ("p0", "critical") and t.status == "open"]
        if p0_tickets:
            issues.append({
                "severity": "critical",
                "title": "P0 Outage Detected",
                "description": f"{len(p0_tickets)} critical open ticket(s): {', '.join(t.subject for t in p0_tickets)}",
                "affected_accounts": list(set(t.account_id for t in p0_tickets)),
                "evidence_ids": [t.ticket_id for t in p0_tickets],
                "suggested_action": {
                    "action_type": "create_escalation",
                    "payload": {
                        "priority": "critical",
                        "reason": "P0 system outage requiring immediate attention",
                        "ticket_id": p0_tickets[0].ticket_id,
                        "account_id": p0_tickets[0].account_id,
                    },
                },
            })

        # 2. Recurring Product Issue (CSV/Bulk Upload cluster)
        csv_tickets = [t for t in tickets if any(kw in t.description.lower() for kw in ["csv", "bulk", "upload", "row"])]
        if len(csv_tickets) >= 2:
            issues.append({
                "severity": "high",
                "title": "Recurring CSV/Bulk Upload Issue",
                "description": f"{len(csv_tickets)} tickets related to CSV upload failures. Likely known issue KI-208.",
                "affected_accounts": list(set(t.account_id for t in csv_tickets)),
                "evidence_ids": [t.ticket_id for t in csv_tickets],
                "suggested_action": {
                    "action_type": "create_followup_task",
                    "payload": {
                        "description": "Investigate recurring CSV upload failures. Cross-reference with Known Issue KI-208.",
                        "related_id": csv_tickets[0].ticket_id,
                    },
                },
            })

        # 3. Known Issue Match: Tracking Stuck (SwiftShip webhook delay KI-211)
        tracking_tickets = [
            t for t in tickets
            if t.status == "open"
            and any(kw in t.description.lower() for kw in ["still shows booked", "tracking", "stuck", "webhook", "not updated"])
        ]
        if tracking_tickets:
            issues.append({
                "severity": "medium",
                "title": "SwiftShip Webhook Delay (KI-211)",
                "description": "Order status not updating after driver pickup. Known SwiftShip webhook delay up to 20 minutes.",
                "affected_accounts": list(set(t.account_id for t in tracking_tickets)),
                "evidence_ids": [t.ticket_id for t in tracking_tickets],
                "suggested_action": {
                    "action_type": "create_followup_task",
                    "payload": {
                        "description": "Verify with SwiftShip carrier. Known 20-min webhook delay (KI-211).",
                        "related_id": tracking_tickets[0].ticket_id,
                    },
                },
            })

        # 4. Security Incidents (API key / credential exposure)
        security_tickets = [
            t for t in tickets
            if t.status == "open"
            and any(kw in t.description.lower() for kw in ["api key", "credential", "exposed", "security", "leaked", "public channel"])
        ]
        if security_tickets:
            issues.append({
                "severity": "critical",
                "title": "Security Incident: Possible Credential Exposure",
                "description": f"{len(security_tickets)} ticket(s) report potential API key or credential exposure.",
                "affected_accounts": list(set(t.account_id for t in security_tickets)),
                "evidence_ids": [t.ticket_id for t in security_tickets],
                "suggested_action": {
                    "action_type": "create_escalation",
                    "payload": {
                        "priority": "critical",
                        "reason": "Security incident: API key exposure. Immediate key rotation and investigation required.",
                        "ticket_id": security_tickets[0].ticket_id,
                        "account_id": security_tickets[0].account_id,
                    },
                },
            })

        # 5. Carrier Fault — Pending Service Credits
        carrier_fault_orders = [o for o in orders if o.carrier_fault and o.status == "BOOKED" and o.picked_up_at is None]
        if carrier_fault_orders:
            issues.append({
                "severity": "medium",
                "title": "Pending Carrier Fault Service Credits",
                "description": f"{len(carrier_fault_orders)} order(s) with carrier-accepted fault still not picked up. Service credits may be owed.",
                "affected_accounts": list(set(o.account_id for o in carrier_fault_orders)),
                "evidence_ids": [o.order_id for o in carrier_fault_orders],
                "suggested_action": {
                    "action_type": "create_followup_task",
                    "payload": {
                        "description": "Review carrier fault orders for service credit eligibility.",
                        "related_id": carrier_fault_orders[0].order_id,
                    },
                },
            })

        # 6. SLA Risk / Breach
        sla_risk_tickets = []
        for t in tickets:
            if t.status != "open":
                continue

            acct = acct_map.get(t.account_id)
            plan = acct.plan if acct else "Standard"
            sla_minutes = _get_sla_minutes(t.account_id, plan, t.priority)

            created_at = t.created_at.replace(tzinfo=None) if t.created_at else None
            if not created_at:
                continue

            elapsed_minutes = (SNAPSHOT_NAIVE - created_at).total_seconds() / 60.0

            if elapsed_minutes >= sla_minutes * 0.8:  # 80% threshold = at risk
                breach_status = "BREACHED" if elapsed_minutes >= sla_minutes else "AT RISK"
                sla_risk_tickets.append({
                    "ticket": t,
                    "elapsed_min": round(elapsed_minutes, 1),
                    "target_min": sla_minutes,
                    "status": breach_status,
                })

        if sla_risk_tickets:
            breached = [s for s in sla_risk_tickets if s["status"] == "BREACHED"]
            at_risk = [s for s in sla_risk_tickets if s["status"] == "AT RISK"]

            desc_parts = []
            if breached:
                desc_parts.append(f"{len(breached)} BREACHED")
            if at_risk:
                desc_parts.append(f"{len(at_risk)} at risk")

            issues.append({
                "severity": "high" if breached else "medium",
                "title": f"SLA Risk/Breach: {', '.join(desc_parts)}",
                "description": "; ".join(
                    f"{s['ticket'].ticket_id}: {s['elapsed_min']}min elapsed vs {s['target_min']}min target ({s['status']})"
                    for s in sla_risk_tickets
                ),
                "affected_accounts": list(set(s["ticket"].account_id for s in sla_risk_tickets)),
                "evidence_ids": [s["ticket"].ticket_id for s in sla_risk_tickets],
                "suggested_action": {
                    "action_type": "create_escalation",
                    "payload": {
                        "priority": "high",
                        "reason": f"SLA breach/risk on {len(sla_risk_tickets)} ticket(s)",
                        "ticket_id": sla_risk_tickets[0]["ticket"].ticket_id,
                        "account_id": sla_risk_tickets[0]["ticket"].account_id,
                    },
                },
            })

    return issues
