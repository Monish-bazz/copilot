"""
Deterministic planner.

Replaces "LLM decides which tools to call" with "LLM fills in a fixed recipe".

For each known intent we define, in code:
  - which entities must be resolved from the DB
  - which document queries MUST be issued (retrieval is never optional)
  - which calculator must run, and with which DB-derived parameters

Unknown intents fall back to the free-form ReAct loop.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Optional


class Intent(str, Enum):
    CANCELLATION = "cancellation"
    SERVICE_CREDIT = "service_credit"
    SLA = "sla"
    PRODUCT_ISSUE = "product_issue"
    SECURITY = "security"
    ORDER_STATUS = "order_status"
    GENERAL = "general"


#: Intents where answering without a document citation is unacceptable.
POLICY_BEARING = {
    Intent.CANCELLATION,
    Intent.SERVICE_CREDIT,
    Intent.SLA,
    Intent.PRODUCT_ISSUE,
    Intent.SECURITY,
}

#: Intents where past ticket resolutions are relevant context (and often traps).
NEEDS_TICKET_HISTORY = {
    Intent.CANCELLATION,
    Intent.SERVICE_CREDIT,
    Intent.PRODUCT_ISSUE,
}

#: Intents that are meaningless without a resolved record. If every requested
#: record is inaccessible, we stop rather than reasoning about nothing.
ENTITY_CENTRIC = {
    Intent.CANCELLATION,
    Intent.SERVICE_CREDIT,
    Intent.SLA,
    Intent.ORDER_STATUS,
}


# --------------------------------------------------------------------------
# Entity extraction
# --------------------------------------------------------------------------

ORDER_RE = re.compile(r"\bORD[-_ ]?(\d{3,})\b", re.IGNORECASE)
TICKET_RE = re.compile(r"\bTKT[-_ ]?(\d{3,})\b", re.IGNORECASE)
ACCOUNT_RE = re.compile(r"\bACCT[-_ ]?(\d{3,})\b", re.IGNORECASE)

#: Friendly names → account ids. Routing aid only; every lookup still goes
#: through the ACL layer, so naming an account you cannot see yields not_found.
ACCOUNT_ALIASES = {
    "northstar": "ACCT-001",
    "north star": "ACCT-001",
    "lumenworks": "ACCT-002",
    "lumen works": "ACCT-002",
    "beacon": "ACCT-003",
    "beacon retail": "ACCT-003",
    "axis": "ACCT-004",
    "axis labs": "ACCT-004",
}


def extract_entities(query: str) -> dict:
    """Pull order/ticket/account identifiers out of the raw query."""
    q = query or ""
    entities: dict = {"order_ids": [], "ticket_ids": [], "account_ids": []}

    for m in ORDER_RE.finditer(q):
        entities["order_ids"].append(f"ORD-{m.group(1)}")
    for m in TICKET_RE.finditer(q):
        entities["ticket_ids"].append(f"TKT-{m.group(1)}")
    for m in ACCOUNT_RE.finditer(q):
        entities["account_ids"].append(f"ACCT-{m.group(1)}")

    lowered = q.lower()
    for alias, acct in ACCOUNT_ALIASES.items():
        if alias in lowered and acct not in entities["account_ids"]:
            entities["account_ids"].append(acct)

    for key in entities:
        seen = set()
        deduped = []
        for v in entities[key]:
            if v not in seen:
                seen.add(v)
                deduped.append(v)
        entities[key] = deduped

    return entities


# --------------------------------------------------------------------------
# Intent classification (deterministic, no LLM — cheap and unit-testable)
# --------------------------------------------------------------------------

SECURITY_TERMS = [
    "api key", "apikey", "credential", "exposed", "exposure", "leaked", "leak",
    "security incident", "compromis", "secret", "token exposed", "public channel",
]

CREDIT_TERMS = [
    "service credit", "credit amount", "credit owed", "eligible for a credit",
    "compensation", "refund for delay", "sla credit",
]

CANCEL_TERMS = ["cancel", "cancellation"]

SLA_TERMS = [
    "sla", "response target", "support target", "support targets",
    "response time", "breached", "breach", "time to respond", "first response",
]

PRODUCT_TERMS = [
    "csv", "bulk upload", "upload", "row limit", "rows", "workaround",
    "known issue", "ki-", "webhook", "still shows", "portal", "tracking",
    "stuck", "http 500", "500 error", "bug", "defect", "intermittent",
    "not updating", "failing", "fails", "error uploading",
]


def _matches(text: str, terms: list[str]) -> list[str]:
    return [t for t in terms if t in text]


AGGREGATE_PATTERNS = [
    "which accounts", "which customers", "all accounts",
    "show me all", "list all", "list my", "show my",
    "how many", "which of my", "all open", "all tickets",
    "all orders", "near breach", "at risk",
]


def classify_intent(query: str) -> Intent:
    """
    Classify into one of the known intents.

    Order matters: more specific / higher-consequence intents are tested first
    so overlapping vocabulary (e.g. "breach" in both security and SLA questions)
    routes predictably.

    AGGREGATE queries (no specific entity, asking about "all" or "which") go to
    GENERAL so the ReAct loop can use query_tickets/query_orders/detect_issues.
    """
    q = (query or "").lower()

    # Aggregate / list queries should use the ReAct path with its full tool set
    if any(p in q for p in AGGREGATE_PATTERNS) and not (ORDER_RE.search(query) or TICKET_RE.search(query)):
        return Intent.GENERAL

    if _matches(q, SECURITY_TERMS):
        return Intent.SECURITY
    if _matches(q, CREDIT_TERMS):
        return Intent.SERVICE_CREDIT
    if _matches(q, CANCEL_TERMS):
        return Intent.CANCELLATION
    if _matches(q, SLA_TERMS):
        return Intent.SLA
    if _matches(q, PRODUCT_TERMS):
        return Intent.PRODUCT_ISSUE
    if "status" in q and (ORDER_RE.search(q) or TICKET_RE.search(q)):
        return Intent.ORDER_STATUS
    return Intent.GENERAL


# --------------------------------------------------------------------------
# Product sub-topic routing
#
# A single grab-bag query ("plan limits bulk upload rows webhook delay carrier")
# embeds between two unrelated topics and retrieves neither well. Route to the
# specific known-issue area instead.
# --------------------------------------------------------------------------

PRODUCT_SUBTOPICS: list[tuple[str, list[str], list[str]]] = [
    (
        "bulk_upload",
        ["csv", "bulk", "upload", "row", "rows", "import", "spreadsheet"],
        [
            "bulk upload supported rows per CSV plan limit Growth Enterprise",
            "known issue bulk upload large CSV intermittent failure workaround split",
        ],
    ),
    (
        "webhook_status",
        ["webhook", "still shows", "still booked", "portal", "tracking", "stuck",
         "not updating", "status update", "picked up", "pickup confirmation", "driver"],
        [
            "SwiftShip webhook delay pickup confirmation status update minutes",
            "known issue KI-211 carrier webhook delayed status not updating",
            "SwiftShip status BOOKED after pickup 20 minutes delay workaround",
        ],
    ),
    (
        "api_errors",
        ["http 500", "500", "api error", "shipment creation", "failing", "outage", "timeout"],
        [
            "shipment creation API errors HTTP 500 known issue",
            "API rate limits error codes troubleshooting",
        ],
    ),
]


def product_subtopic_queries(raw_query: str) -> list[str]:
    """Targeted known-issue queries based on which product area the question is about."""
    q = (raw_query or "").lower()
    out: list[str] = []
    for _name, triggers, queries in PRODUCT_SUBTOPICS:
        if any(t in q for t in triggers):
            out.extend(queries)
    if not out:
        out.append("known issues product operations guide workaround")
    return out


def product_subtopic_names(raw_query: str) -> list[str]:
    q = (raw_query or "").lower()
    return [name for name, triggers, _ in PRODUCT_SUBTOPICS if any(t in q for t in triggers)]


# --------------------------------------------------------------------------
# Retrieval recipes — these queries are ALWAYS issued for the intent
# --------------------------------------------------------------------------

def retrieval_queries(intent: Intent, account_name: str, raw_query: str) -> list[str]:
    """
    Templated document queries per intent. One query biases toward the customer's
    own agreement; the rest target global policy so the resolver always has both
    sides of a potential override to compare.
    """
    name = account_name or "customer"

    if intent is Intent.CANCELLATION:
        return [
            f"{name} agreement cancellation fee waiver terms",
            "cancellation fee free window 30 minutes standard SOP",
            "cancellation before pickup after pickup fee amount INR",
        ]

    if intent is Intent.SERVICE_CREDIT:
        return [
            f"{name} agreement service credit terms delay threshold",
            "service credit carrier fault pickup delay eligibility",
            "service credit calculation cap percentage shipment fee INR",
        ]

    if intent is Intent.SLA:
        return [
            f"{name} agreement SLA response targets priority P1",
            "support SLA response time targets priority P0 P1 P2",
            "premium support enterprise response commitments",
        ]

    if intent is Intent.PRODUCT_ISSUE:
        # raw query first, then narrowly-targeted sub-topic queries
        return [raw_query] + product_subtopic_queries(raw_query)

    if intent is Intent.SECURITY:
        return [
            "security incident escalation immediate procedure",
            "API key exposure credential compromise rotate revoke",
            "escalation severity matrix support policy",
        ]

    if intent is Intent.ORDER_STATUS:
        return []

    return [raw_query]


def broadened_queries(intent: Intent, raw_query: str) -> list[str]:
    """Fallback queries used by the evidence gate when the first pass found nothing."""
    base = [raw_query, "support policy", "cancellation and service credit SOP"]
    if intent is Intent.PRODUCT_ISSUE:
        base.append("product operations guide known issues")
    if intent is Intent.SLA:
        base.append("service level agreement response targets")
    if intent is Intent.SECURITY:
        base.append("security escalation policy")
    return base


# --------------------------------------------------------------------------
# Ticket history relevance
# --------------------------------------------------------------------------

HISTORY_RELEVANCE: dict[Intent, list[str]] = {
    Intent.CANCELLATION: ["cancel", "cancellation", "fee", "refund", "waiv"],
    Intent.SERVICE_CREDIT: ["credit", "delay", "carrier", "compensat", "refund"],
    Intent.PRODUCT_ISSUE: [
        "csv", "bulk", "upload", "row", "webhook", "tracking", "portal",
        "status", "500", "api", "limit", "plan",
    ],
}


def history_is_relevant(intent: Intent, ticket: dict) -> bool:
    """
    Only attach a past ticket resolution when it is topically related.

    Without this, a cancellation-fee ticket (TKT-450) gets injected into a
    webhook question and the composer dutifully narrates it as a conflict.
    """
    terms = HISTORY_RELEVANCE.get(intent)
    if not terms:
        return False
    blob = " ".join(
        str(ticket.get(k, "") or "")
        for k in ("subject", "description", "resolution", "category")
    ).lower()
    return any(t in blob for t in terms)


# --------------------------------------------------------------------------
# Contract term extraction (facts come from the retrieved documents)
# --------------------------------------------------------------------------

#: Neutral. No account's real terms appear here — a missing contract must never
#: supply numbers that look like findings. Numeric fields stay None until a
#: contract actually states them; the calculator applies standard policy instead.
CAPABILITY_DEFAULTS = {
    "contract_found": False,
    "has_cancellation_waiver": False,
    "cancellation_defers_to_standard": False,
    "has_custom_credit_rule": False,
    "credit_threshold_hours": None,
    "credit_fixed_inr": None,
    "custom_sla_p1_minutes": None,
    "notes": "",
}


WAIVER_PATTERNS = [
    r"waiv\w*[^.]{0,80}cancellation",
    r"cancellation[^.]{0,80}waiv\w*",
    r"no cancellation fee",
    r"without (?:a )?cancellation fee",
    r"cancellation fee (?:is |shall be )?(?:not applicable|nil|zero|inr\s*0)",
]

DEFERS_PATTERNS = [
    r"standard[^.]{0,60}cancellation[^.]{0,60}(?:polic|appl|terms)",
    r"cancellation[^.]{0,60}standard[^.]{0,60}(?:polic|appl|terms)",
    r"standard polic\w*[^.]{0,60}appl",
    r"standard (?:cancellation )?(?:policy|policies|terms) (?:shall |will |to )?appl",
    r"subject to[^.]{0,40}standard[^.]{0,40}(?:polic|cancel|terms)",
    r"governed by[^.]{0,40}standard[^.]{0,40}(?:polic|cancel|terms)",
    r"as per[^.]{0,40}standard[^.]{0,40}(?:polic|cancel|terms)",
    r"follow[^.]{0,40}standard[^.]{0,40}(?:polic|cancel|terms)",
    r"(?:cancellation|cancel)[^.]{0,40}(?:as per|per|follow|under|governed by|subject to)[^.]{0,40}standard",
    r"no (?:special|custom|specific)[^.]{0,40}cancellation",
    r"cancellation[^.]{0,40}(?:default|general|standard)[^.]{0,40}(?:rule|polic|term|sop)",
]


def extract_capabilities_regex(contract_text: str) -> dict:
    """
    Deterministic extraction of contract terms from the account's own contract
    chunks. Also the fallback when LLM extraction is unavailable or unparseable.
    """
    caps = dict(CAPABILITY_DEFAULTS)
    if not contract_text or not contract_text.strip():
        return caps

    caps["contract_found"] = True
    text = contract_text.lower()

    for pat in DEFERS_PATTERNS:
        if re.search(pat, text):
            caps["cancellation_defers_to_standard"] = True
            break

    if not caps["cancellation_defers_to_standard"]:
        for pat in WAIVER_PATTERNS:
            if re.search(pat, text):
                caps["has_cancellation_waiver"] = True
                break

    credit_ctx = re.search(r"(?:service )?credit[^.]{0,160}", text)
    if credit_ctx:
        seg = credit_ctx.group(0)
        amount = re.search(r"inr\s*([\d,]+)", seg)
        threshold = re.search(
            r"(?:more than|exceed\w*|greater than|over|beyond|>)\s*([\d.]+)\s*hour", seg
        )
        if amount and threshold:
            caps["has_custom_credit_rule"] = True
            caps["credit_fixed_inr"] = float(amount.group(1).replace(",", ""))
            caps["credit_threshold_hours"] = float(threshold.group(1))

    sla_match = re.search(r"p1[^.]{0,80}?(\d+)\s*minute", text)
    if sla_match:
        caps["custom_sla_p1_minutes"] = float(sla_match.group(1))

    return caps


def sanitize_capabilities(caps: dict) -> dict:
    """
    Enforce internal consistency so downstream code and the composer cannot be
    misled by a partially-extracted contract.
    """
    out = dict(CAPABILITY_DEFAULTS)
    out.update(caps or {})

    # Deferring to standard policy is mutually exclusive with a waiver
    if out.get("cancellation_defers_to_standard"):
        out["has_cancellation_waiver"] = False

    # A custom credit rule is only real if BOTH threshold and amount were found
    if out.get("has_custom_credit_rule"):
        if out.get("credit_threshold_hours") is None or out.get("credit_fixed_inr") is None:
            out["has_custom_credit_rule"] = False

    # No contract means nothing can have been extracted from one
    if not out.get("contract_found"):
        out["has_cancellation_waiver"] = False
        out["cancellation_defers_to_standard"] = False
        out["has_custom_credit_rule"] = False
        out["credit_threshold_hours"] = None
        out["credit_fixed_inr"] = None
        out["custom_sla_p1_minutes"] = None

    return out


CAPABILITY_EXTRACTION_PROMPT = """You are reading excerpts from ONE customer's service agreement.
Extract only what the text explicitly states. Do not infer, and do not use outside knowledge.

Return strict JSON with exactly these keys:
{
  "has_cancellation_waiver": bool,           // the contract waives the cancellation fee
  "cancellation_defers_to_standard": bool,   // the contract says standard cancellation policy applies
  "has_custom_credit_rule": bool,            // the contract defines its own service credit
  "credit_threshold_hours": number or null,  // delay hours required for that custom credit
  "credit_fixed_inr": number or null,        // fixed custom credit amount in INR
  "custom_sla_p1_minutes": number or null,   // custom P1 response target in MINUTES
  "notes": string                            // one short sentence quoting the key clause
}

If a field is not stated in the excerpts, use false or null. Never guess a number.
Output ONLY the JSON object.

CONTRACT EXCERPTS:
"""
