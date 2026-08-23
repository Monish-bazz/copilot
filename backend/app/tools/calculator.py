"""
Deterministic calculator tools.
All time comparisons use SNAPSHOT_TS, never wall-clock.
"""
from datetime import datetime
from app.config import SNAPSHOT_TS


def _to_naive(iso: str) -> datetime:
    """Parse an ISO timestamp and strip timezone to compare with naive snapshot."""
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    # Strip tz info; all data is IST and snapshot is IST
    return dt.replace(tzinfo=None)


SNAPSHOT_NAIVE = SNAPSHOT_TS.replace(tzinfo=None)


def get_snapshot_time() -> str:
    """Return the snapshot timestamp as ISO string."""
    return SNAPSHOT_TS.isoformat()


def hours_between(start_iso: str, end_iso: str) -> float:
    """Calculate hours between two ISO timestamps."""
    try:
        start = _to_naive(start_iso)
        end = _to_naive(end_iso)
        diff = end - start
        return round(diff.total_seconds() / 3600.0, 2)
    except Exception:
        return -1.0


def minutes_since_booking(booked_at_iso: str) -> float:
    """Minutes elapsed from booked_at to the snapshot time."""
    try:
        dt = _to_naive(booked_at_iso)
        diff = SNAPSHOT_NAIVE - dt
        return round(diff.total_seconds() / 60.0, 2)
    except Exception:
        return -1.0


def minutes_between(start_iso: str, end_iso: str) -> float:
    """Minutes between two ISO timestamps."""
    try:
        start = _to_naive(start_iso)
        end = _to_naive(end_iso)
        diff = end - start
        return round(diff.total_seconds() / 60.0, 2)
    except Exception:
        return -1.0


def pickup_delay_hours(pickup_window_end_iso: str) -> float:
    """
    Hours the pickup is overdue relative to the snapshot.
    Returns negative if still within window.
    """
    try:
        window_end = _to_naive(pickup_window_end_iso)
        diff = SNAPSHOT_NAIVE - window_end
        return round(diff.total_seconds() / 3600.0, 2)
    except Exception:
        return -1.0


def sla_remaining_minutes(created_at_iso: str, sla_minutes: float) -> float:
    """
    Calculate SLA minutes remaining for a ticket.
    Negative = breached.
    """
    try:
        dt = _to_naive(created_at_iso)
        elapsed_minutes = (SNAPSHOT_NAIVE - dt).total_seconds() / 60.0
        return round(sla_minutes - elapsed_minutes, 2)
    except Exception:
        return -1.0


def sla_remaining_hours(created_at_iso: str, sla_hours: float) -> float:
    """
    Calculate SLA hours remaining for a ticket.
    Negative = breached.
    """
    return round(sla_remaining_minutes(created_at_iso, sla_hours * 60.0) / 60.0, 4)


def cancellation_fee(minutes_since_booking: float, has_custom_waiver: bool, status: str) -> dict:
    """
    Determine cancellation fee based on standard SOP v4 rules.
    
    Standard rules:
    - Within 30 minutes of booking AND not yet picked up: FREE
    - After 30 minutes AND not yet picked up: INR 250
    - After pickup: INR 500

    If has_custom_waiver is True (contract overrides), fee is 0 for any BOOKED order.
    
    Args:
        minutes_since_booking: minutes elapsed since booking to cancellation request
        has_custom_waiver: whether the account's contract waives the fee
        status: order status (BOOKED, PICKED_UP, DELIVERED, etc.)
    """
    if status == "DELIVERED":
        return {
            "fee_inr": 0,
            "cancellable": False,
            "reason": "Order already delivered. Cancellation not possible.",
        }

    if status == "PICKED_UP":
        if has_custom_waiver:
            return {
                "fee_inr": 500,
                "cancellable": True,
                "reason": "Contract waiver applies only to BOOKED shipments before pickup. This order is already PICKED_UP, so the standard post-pickup fee of INR 500 applies.",
            }
        return {
            "fee_inr": 500,
            "cancellable": True,
            "reason": "Standard policy: INR 500 fee for cancellation after pickup.",
        }

    # BOOKED (not picked up)
    if has_custom_waiver:
        return {
            "fee_inr": 0,
            "cancellable": True,
            "reason": "No cancellation fee. Customer contract waives fee for BOOKED shipments before pickup.",
        }

    if minutes_since_booking <= 30:
        return {
            "fee_inr": 0,
            "cancellable": True,
            "reason": "Free cancellation: within 30-minute window after booking.",
        }

    return {
        "fee_inr": 250,
        "cancellable": True,
        "reason": "Standard policy: INR 250 fee for cancellation after 30 minutes (before pickup).",
    }


def service_credit(
    delay_hours: float,
    carrier_fault: bool,
    shipment_fee_inr: float,
    has_custom_credit_rule: bool = False,
    custom_threshold_hours: float = 4.0,
    custom_fixed_credit_inr: float = 300.0,
) -> dict:
    """
    Determine service credit eligibility.

    Standard SOP v4 rules:
    - Carrier must accept fault
    - Delay > 2 hours past pickup window end
    - Credit = min(INR 500, 10% of shipment fee)

    Custom contract (LumenWorks):
    - Delay > custom_threshold_hours (default 4h)
    - Fixed credit = custom_fixed_credit_inr (default INR 300)
    """
    if not carrier_fault:
        return {
            "eligible": False,
            "credit_inr": 0,
            "reason": "Carrier has not accepted fault. No credit applicable.",
        }

    if has_custom_credit_rule:
        if delay_hours > custom_threshold_hours:
            return {
                "eligible": True,
                "credit_inr": custom_fixed_credit_inr,
                "reason": f"Contract-specific credit: fixed INR {custom_fixed_credit_inr} for delay > {custom_threshold_hours}h (actual delay: {delay_hours}h).",
            }
        else:
            return {
                "eligible": False,
                "credit_inr": 0,
                "reason": f"Contract threshold not met. Delay {delay_hours}h < required {custom_threshold_hours}h.",
            }

    # Standard SOP
    if delay_hours > 2.0:
        standard_credit = min(500.0, shipment_fee_inr * 0.10)
        return {
            "eligible": True,
            "credit_inr": standard_credit,
            "reason": f"Standard SOP credit: min(INR 500, 10% of INR {shipment_fee_inr}) = INR {standard_credit}. Delay: {delay_hours}h > 2h threshold.",
        }
    else:
        return {
            "eligible": False,
            "credit_inr": 0,
            "reason": f"Standard SOP: delay {delay_hours}h does not exceed 2h threshold.",
        }


def calculate_fee(base_amount: float, rule: str) -> dict:
    """Legacy fee calculator kept for backward compatibility."""
    if rule == "free":
        return {"fee": 0.0, "waived": True, "reason": "Fee waived by policy or contract."}
    elif rule == "standard_cancel_after_30":
        return {"fee": 250.0, "waived": False, "reason": "Standard cancellation fee INR 250 (after 30 min)."}
    elif rule == "standard_cancel_after_pickup":
        return {"fee": 500.0, "waived": False, "reason": "Standard cancellation fee INR 500 (after pickup)."}
    elif rule == "percentage_10":
        return {"fee": round(base_amount * 0.1, 2), "waived": False, "reason": "10% of shipment fee."}
    return {"fee": 0.0, "waived": False, "reason": "Unknown rule."}
