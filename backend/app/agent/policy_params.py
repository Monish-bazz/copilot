"""
Policy Parameter Store (Phase 3).

Extracts concrete numbers from the retrieved policy/SOP documents so that
calculators use document-derived figures, not compiled constants.

Extraction is regex-based for speed and determinism. The store caches by
source_id so repeated calls within a turn are cheap.

If extraction fails (the expected text isn't found in the retrieved chunks),
the store falls back to hardcoded defaults — but those defaults are labelled
as fallbacks in the provenance, so the answer can say "standard SOP (assumed)"
instead of citing a specific page.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PolicyParameters:
    """Extracted from the global policy + SOP corpus."""
    # Cancellation
    free_window_minutes: float = 30.0
    fee_after_window_inr: float = 250.0
    fee_after_pickup_inr: float = 500.0
    free_window_source: str = "default (SOP v4 assumed)"
    fee_after_window_source: str = "default (SOP v4 assumed)"
    fee_after_pickup_source: str = "default (SOP v4 assumed)"

    # Service Credit
    credit_delay_threshold_hours: float = 2.0
    credit_cap_inr: float = 500.0
    credit_pct_of_fee: float = 0.10
    credit_threshold_source: str = "default (SOP v4 assumed)"
    credit_cap_source: str = "default (SOP v4 assumed)"

    # SLA (standard plan defaults)
    sla_enterprise_p0_minutes: float = 15.0
    sla_enterprise_p1_minutes: float = 30.0
    sla_enterprise_p2_minutes: float = 120.0
    sla_growth_p0_minutes: float = 30.0
    sla_growth_p1_minutes: float = 120.0
    sla_growth_p2_minutes: float = 240.0
    sla_standard_p0_minutes: float = 60.0
    sla_standard_p1_minutes: float = 240.0
    sla_standard_p2_minutes: float = 1440.0
    sla_source: str = "default (Support Policy v3 assumed)"

    # Provenance
    extracted_from: list = field(default_factory=list)


# Regex extractors for SOP/policy text
_FREE_WINDOW = re.compile(
    r"(?:within|first|free\s*(?:cancellation)?\s*window)[^.]{0,40}?(\d+)\s*min",
    re.IGNORECASE,
)
_FEE_AFTER_WINDOW = re.compile(
    r"(?:after|beyond|exceed)[^.]{0,60}?(?:inr|₹)\s*([\d,]+)[^.]{0,30}(?:before|not\s*(?:yet\s*)?picked)",
    re.IGNORECASE,
)
_FEE_AFTER_PICKUP = re.compile(
    r"(?:after\s*pickup|picked.up|post.pickup)[^.]{0,60}?(?:inr|₹)\s*([\d,]+)",
    re.IGNORECASE,
)
_CREDIT_THRESHOLD = re.compile(
    r"(?:service\s*credit|credit\s*eligib)[^.]{0,100}?(?:exceed|more\s*than|greater\s*than|over|beyond|>)\s*([\d.]+)\s*hour",
    re.IGNORECASE,
)
_CREDIT_CAP = re.compile(
    r"(?:cap|maximum|ceiling|up\s*to|lower\s*of)[^.]{0,60}?(?:inr|₹)\s*([\d,]+)",
    re.IGNORECASE,
)
_CREDIT_PCT = re.compile(
    r"(\d+)(?:\s*%|\s*percent)[^.]{0,40}(?:shipment|fee|value)",
    re.IGNORECASE,
)
_SLA_TARGET = re.compile(
    r"(p[012]|priority\s*[012])[^.]{0,60}?(\d+)\s*(minute|hour)",
    re.IGNORECASE,
)


def extract_policy_params(citations: list[dict]) -> PolicyParameters:
    """
    Extract policy parameters from the global-scope current citations.

    Only reads chunks from the global policy tier (POL-V3, SOP-V4). Contract
    chunks are NOT used here — contract overrides are handled in capabilities.
    """
    params = PolicyParameters()

    # Collect text from global current sources only
    global_text_parts: list[str] = []
    sources_used: list[str] = []

    for c in citations:
        if c.get("scope") != "global":
            continue
        if c.get("status") != "current":
            continue
        if (c.get("authority") or 0) < 50:
            continue
        global_text_parts.append(c.get("excerpt", ""))
        sid = c.get("source_id", "")
        if sid and sid not in sources_used:
            sources_used.append(sid)

    if not global_text_parts:
        return params

    text = "\n\n".join(global_text_parts).lower()
    params.extracted_from = sources_used

    # Cancellation
    m = _FREE_WINDOW.search(text)
    if m:
        params.free_window_minutes = float(m.group(1))
        params.free_window_source = f"extracted from {sources_used}"

    m = _FEE_AFTER_WINDOW.search(text)
    if m:
        params.fee_after_window_inr = float(m.group(1).replace(",", ""))
        params.fee_after_window_source = f"extracted from {sources_used}"

    m = _FEE_AFTER_PICKUP.search(text)
    if m:
        params.fee_after_pickup_inr = float(m.group(1).replace(",", ""))
        params.fee_after_pickup_source = f"extracted from {sources_used}"

    # Service Credit
    m = _CREDIT_THRESHOLD.search(text)
    if m:
        params.credit_delay_threshold_hours = float(m.group(1))
        params.credit_threshold_source = f"extracted from {sources_used}"

    m = _CREDIT_CAP.search(text)
    if m:
        params.credit_cap_inr = float(m.group(1).replace(",", ""))
        params.credit_cap_source = f"extracted from {sources_used}"

    m = _CREDIT_PCT.search(text)
    if m:
        params.credit_pct_of_fee = float(m.group(1)) / 100.0

    # SLA targets
    for m in _SLA_TARGET.finditer(text):
        priority_raw = m.group(1).lower().replace("priority", "").strip()
        value = float(m.group(2))
        unit = m.group(3).lower()
        if "hour" in unit:
            value *= 60.0  # convert to minutes

        # Try to detect which plan this applies to from surrounding context
        # This is best-effort; the defaults are reasonable
        start = max(0, m.start() - 200)
        context = text[start:m.end() + 50]
        if "enterprise" in context:
            if priority_raw in ("0", "p0"):
                params.sla_enterprise_p0_minutes = value
            elif priority_raw in ("1", "p1"):
                params.sla_enterprise_p1_minutes = value
            elif priority_raw in ("2", "p2"):
                params.sla_enterprise_p2_minutes = value
            params.sla_source = f"extracted from {sources_used}"
        elif "growth" in context:
            if priority_raw in ("0", "p0"):
                params.sla_growth_p0_minutes = value
            elif priority_raw in ("1", "p1"):
                params.sla_growth_p1_minutes = value
            elif priority_raw in ("2", "p2"):
                params.sla_growth_p2_minutes = value
            params.sla_source = f"extracted from {sources_used}"
        elif "standard" in context:
            if priority_raw in ("0", "p0"):
                params.sla_standard_p0_minutes = value
            elif priority_raw in ("1", "p1"):
                params.sla_standard_p1_minutes = value
            elif priority_raw in ("2", "p2"):
                params.sla_standard_p2_minutes = value
            params.sla_source = f"extracted from {sources_used}"

    return params


def get_sla_target_minutes(params: PolicyParameters, plan: str, priority: str, custom_p1_minutes: Optional[float] = None) -> tuple[float, str]:
    """
    Get SLA target in minutes, preferring contract-specific override, then extracted params.

    Returns (target_minutes, source_description).
    """
    priority = (priority or "").lower()
    if priority in ("critical",):
        priority = "p0"

    # Contract override takes absolute precedence
    if custom_p1_minutes and priority in ("p0", "p1"):
        return (custom_p1_minutes, "customer agreement (custom SLA)")

    plan_lower = (plan or "standard").lower()
    if "enterprise" in plan_lower:
        targets = {
            "p0": params.sla_enterprise_p0_minutes,
            "p1": params.sla_enterprise_p1_minutes,
            "p2": params.sla_enterprise_p2_minutes,
        }
    elif "growth" in plan_lower:
        targets = {
            "p0": params.sla_growth_p0_minutes,
            "p1": params.sla_growth_p1_minutes,
            "p2": params.sla_growth_p2_minutes,
        }
    else:
        targets = {
            "p0": params.sla_standard_p0_minutes,
            "p1": params.sla_standard_p1_minutes,
            "p2": params.sla_standard_p2_minutes,
        }

    target = targets.get(priority, 2880.0)
    return (target, f"{plan} plan SLA ({params.sla_source})")
