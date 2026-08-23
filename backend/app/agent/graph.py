"""
ParcelPilot Copilot agent graph.

Two execution paths:

  1. RECIPE PATH (known intents) — deterministic. The code, not the model,
     decides that documents get retrieved and that the calculator runs.
     classify -> entities -> retrieve -> capabilities -> compute
              -> resolve -> evidence_gate -> [action -> confirm] -> compose

  2. REACT PATH (Intent.GENERAL) — the original free-form tool loop, kept as a
     fallback for questions outside the known intents.

Tools are invoked as Runnables (`tool.invoke(args, config)`) rather than called
directly, so `astream_events` still emits on_tool_start/on_tool_end and the UI
tool chips keep working on the deterministic path.
"""
from __future__ import annotations

import json
from typing import Annotated, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from app.config import NIM_MODEL, NVIDIA_API_KEY, NIM_BASE_URL
from app.agent.prompts import SYSTEM_PROMPT, COMPOSE_PROMPT, REACT_SYSTEM_PROMPT, DENIAL_PROMPT
from app.agent.answer_schema import parse_answer, StructuredAnswer
from app.agent.resolver import resolve_and_score
from app.agent.planner import (
    Intent,
    classify_intent,
    extract_entities,
    retrieval_queries,
    broadened_queries,
    extract_capabilities_regex,
    sanitize_capabilities,
    history_is_relevant,
    CAPABILITY_DEFAULTS,
    CAPABILITY_EXTRACTION_PROMPT,
    POLICY_BEARING,
    NEEDS_TICKET_HISTORY,
    ENTITY_CENTRIC,
)

from app.tools.documents import search_documents as _search_documents
from app.tools.lookup import (
    get_account as _get_account,
    get_order as _get_order,
    get_ticket as _get_ticket,
    query_orders as _query_orders,
    query_tickets as _query_tickets,
)
from app.tools.calculator import (
    hours_between as _hours_between,
    minutes_since_booking as _minutes_since_booking,
    minutes_between as _minutes_between,
    pickup_delay_hours as _pickup_delay_hours,
    sla_remaining_hours as _sla_remaining_hours,
    sla_remaining_minutes as _sla_remaining_minutes,
    cancellation_fee as _cancellation_fee,
    service_credit as _service_credit,
    calculate_fee as _calculate_fee,
)
from app.tools.actions import prepare_action as _prepare_action, execute_action as _execute_action
from app.ops.detectors import run_detectors as _run_detectors
from app.agent.policy_params import extract_policy_params, get_sla_target_minutes


# ==========================================================================
# State
# ==========================================================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_context: dict
    original_query: str

    intent: Optional[str]
    entities: Optional[dict]
    records: Optional[dict]          # resolved DB rows: account/order/ticket/history
    capabilities: Optional[dict]     # contract terms extracted from retrieved chunks
    computed: Optional[dict]         # calculator output
    plan_trace: Optional[list]       # human-readable recipe steps executed

    citations: list
    resolution: Optional[dict]
    retrieval_attempt: int

    pending_draft: Optional[dict]
    user_confirmation: Optional[dict]
    execute_result: Optional[dict]

    iterations: int
    final_answer: Optional[str]


# ==========================================================================
# Tools
# ==========================================================================
@tool
def search_documents(query: str, include_deprecated: bool = False, config: RunnableConfig = None) -> list[dict]:
    """Search authoritative documents (policies, SOPs, contracts). Returns chunks with metadata."""
    ctx = (config or {}).get("configurable", {}).get("user_context")
    return _search_documents(query, include_deprecated, ctx)


@tool
def get_account(account_id: str, config: RunnableConfig = None) -> dict:
    """Get account details by account_id."""
    ctx = (config or {}).get("configurable", {}).get("user_context")
    return _get_account(account_id, ctx)


@tool
def get_order(order_id: str, config: RunnableConfig = None) -> dict:
    """Get order details including timestamps, fees, carrier_fault and notes."""
    ctx = (config or {}).get("configurable", {}).get("user_context")
    return _get_order(order_id, ctx)


@tool
def get_ticket(ticket_id: str, config: RunnableConfig = None) -> dict:
    """Get ticket details including description, historical resolution, priority and SLA."""
    ctx = (config or {}).get("configurable", {}).get("user_context")
    return _get_ticket(ticket_id, ctx)


@tool
def query_orders(status: str = None, account_id: str = None, config: RunnableConfig = None) -> list[dict]:
    """Query orders with optional status/account filters."""
    ctx = (config or {}).get("configurable", {}).get("user_context")
    return _query_orders(status, account_id, ctx)


@tool
def query_tickets(status: str = None, priority: str = None, account_id: str = None, config: RunnableConfig = None) -> list[dict]:
    """Query tickets with optional status/priority/account filters."""
    ctx = (config or {}).get("configurable", {}).get("user_context")
    return _query_tickets(status, priority, account_id, ctx)


@tool
def minutes_since_booking(booked_at_iso: str) -> float:
    """Minutes elapsed from booked_at to the snapshot time (2026-08-16 11:00 IST)."""
    return _minutes_since_booking(booked_at_iso)


@tool
def hours_between(start_iso: str, end_iso: str) -> float:
    """Hours between two ISO timestamps."""
    return _hours_between(start_iso, end_iso)


@tool
def minutes_between(start_iso: str, end_iso: str) -> float:
    """Minutes between two ISO timestamps."""
    return _minutes_between(start_iso, end_iso)


@tool
def pickup_delay_hours(pickup_window_end_iso: str) -> float:
    """Hours the pickup is overdue relative to the snapshot. Negative = still within window."""
    return _pickup_delay_hours(pickup_window_end_iso)


@tool
def sla_remaining_minutes(created_at_iso: str, sla_minutes: float) -> float:
    """SLA minutes remaining. Negative means breached."""
    return _sla_remaining_minutes(created_at_iso, sla_minutes)


@tool
def sla_remaining_hours(created_at_iso: str, sla_hours: float) -> float:
    """SLA hours remaining. Negative means breached."""
    return _sla_remaining_hours(created_at_iso, sla_hours)


@tool
def cancellation_fee(minutes_since_booking: float, has_custom_waiver: bool, status: str) -> dict:
    """Calculate the cancellation fee from elapsed minutes, contract waiver flag and order status."""
    return _cancellation_fee(minutes_since_booking, has_custom_waiver, status)


@tool
def service_credit(
    delay_hours: float,
    carrier_fault: bool,
    shipment_fee_inr: float,
    has_custom_credit_rule: bool = False,
    custom_threshold_hours: float = 4.0,
    custom_fixed_credit_inr: float = 300.0,
) -> dict:
    """Calculate service credit eligibility and amount."""
    return _service_credit(
        delay_hours, carrier_fault, shipment_fee_inr,
        has_custom_credit_rule, custom_threshold_hours, custom_fixed_credit_inr,
    )


@tool
def calculate_fee(base_amount: float, rule: str) -> dict:
    """Legacy fee helper. Rules: free, standard_cancel_after_30, standard_cancel_after_pickup, percentage_10."""
    return _calculate_fee(base_amount, rule)


@tool
def prepare_action(action_type: str, payload: dict, config: RunnableConfig = None) -> dict:
    """Prepare a state-changing action draft (create_escalation, create_followup_task)."""
    ctx = (config or {}).get("configurable", {}).get("user_context")
    return _prepare_action(action_type, payload, ctx)


@tool
def detect_issues(config: RunnableConfig = None) -> list[dict]:
    """Run proactive issue detectors. Returns detected issues across all accounts. Internal only."""
    ctx = (config or {}).get("configurable", {}).get("user_context")
    role = ctx.get("role", "") if isinstance(ctx, dict) else getattr(ctx, "role", "")
    if role == "customer":
        return [{"error": "access_denied", "message": "Internal users only."}]
    return _run_detectors(ctx)


tools = [
    search_documents, get_account, get_order, get_ticket,
    query_orders, query_tickets,
    minutes_since_booking, hours_between, minutes_between,
    pickup_delay_hours, sla_remaining_minutes, sla_remaining_hours,
    cancellation_fee, service_credit, calculate_fee,
    prepare_action, detect_issues,
]

tool_node = ToolNode(tools)


# ==========================================================================
# LLMs
# ==========================================================================
llm = ChatNVIDIA(
    model=NIM_MODEL, nvidia_api_key=NVIDIA_API_KEY, base_url=NIM_BASE_URL
).bind_tools(tools)

# Unbound: used for capability extraction and final composition so the model
# cannot answer a "give me JSON" request with a tool call.
plain_llm = ChatNVIDIA(
    model=NIM_MODEL, nvidia_api_key=NVIDIA_API_KEY, base_url=NIM_BASE_URL
)


# ==========================================================================
# Helpers
# ==========================================================================
def _ctx_of(state: AgentState):
    return state.get("user_context") or {}


def _trace(state: AgentState, msg: str) -> list:
    trace = list(state.get("plan_trace") or [])
    trace.append(msg)
    return trace


def _parse_json_loose(text: str) -> Optional[dict]:
    """Parse JSON that may be wrapped in markdown fences or trailing prose."""
    if not text:
        return None
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _ok(rec) -> bool:
    """True if a tool result is a usable record (not an error envelope)."""
    return isinstance(rec, dict) and not rec.get("error")


# ==========================================================================
# RECIPE PATH NODES
# ==========================================================================
def classify(state: AgentState) -> dict:
    """Deterministic intent + entity extraction. No LLM, no tool calls."""
    query = state.get("original_query") or ""
    if not query:
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage):
                query = m.content
                break

    intent = classify_intent(query)
    entities = extract_entities(query)

    return {
        "original_query": query,
        "intent": intent.value,
        "entities": entities,
        "plan_trace": [f"intent={intent.value}", f"entities={entities}"],
        "retrieval_attempt": 0,
        "citations": [],
        "records": {},
        "capabilities": None,
        "computed": None,
    }


def gather_entities(state: AgentState, config: RunnableConfig) -> dict:
    """
    Resolve every identifier found in the query against the DB, through the
    ACL-enforcing tools. Also derives the SUBJECT account (which may differ
    from the logged-in user's account when an internal agent is asking).
    """
    intent = Intent(state.get("intent") or Intent.GENERAL.value)
    entities = state.get("entities") or {}
    records: dict = {
        "orders": {}, "tickets": {}, "account": None, "history": [],
        "denied": [], "resolved_any": False,
    }

    subject_account_id = None
    requested = 0

    for order_id in entities.get("order_ids", []):
        requested += 1
        rec = get_order.invoke({"order_id": order_id}, config=config)
        if _ok(rec):
            records["orders"][order_id] = rec
            records["resolved_any"] = True
            if not subject_account_id:
                subject_account_id = rec.get("account_id")
        else:
            records["denied"].append(order_id)

    for ticket_id in entities.get("ticket_ids", []):
        requested += 1
        rec = get_ticket.invoke({"ticket_id": ticket_id}, config=config)
        if _ok(rec):
            records["tickets"][ticket_id] = rec
            records["resolved_any"] = True
            if not subject_account_id:
                subject_account_id = rec.get("account_id")
        else:
            records["denied"].append(ticket_id)

    # Explicit account mention, or inferred from the order/ticket
    if not subject_account_id:
        for acct in entities.get("account_ids", []):
            subject_account_id = acct
            break

    ctx = _ctx_of(state)
    user_account_id = ctx.get("account_id") if isinstance(ctx, dict) else getattr(ctx, "account_id", None)
    user_role = ctx.get("role", "") if isinstance(ctx, dict) else getattr(ctx, "role", "")

    if not subject_account_id:
        subject_account_id = user_account_id

    if subject_account_id:
        acct = get_account.invoke({"account_id": subject_account_id}, config=config)
        records["account"] = acct if _ok(acct) else None

    # Pull ticket history for intents where past (possibly wrong) advice matters.
    # Filtered by topical relevance so a cancellation ticket does not get
    # attached to a webhook question.
    if intent in NEEDS_TICKET_HISTORY and subject_account_id:
        hist = query_tickets.invoke({"account_id": subject_account_id}, config=config)
        if isinstance(hist, list):
            records["history"] = [
                h for h in hist
                if _ok(h) and h.get("resolution") and history_is_relevant(intent, h)
            ]

    records["subject_account_id"] = subject_account_id

    # Access blocked: the user asked about specific records, every one of them was
    # refused, and this intent is meaningless without a record. Stop here so we do
    # not reason (or narrate) about data the user cannot see.
    access_blocked = (
        requested > 0
        and not records["resolved_any"]
        and intent in ENTITY_CENTRIC
    )
    # A customer naming an account that is not theirs is always a denial.
    if (
        user_role == "customer"
        and subject_account_id
        and user_account_id
        and subject_account_id != user_account_id
    ):
        access_blocked = True

    records["access_blocked"] = access_blocked
    if access_blocked:
        # Do not retain any cross-account identifiers for the composer.
        records["subject_account_id"] = None
        records["account"] = None
        records["history"] = []

    trace = f"resolved subject_account={records['subject_account_id']}"
    if records["denied"]:
        trace += f", inaccessible={len(records['denied'])}"
    if access_blocked:
        trace += ", ACCESS BLOCKED -> short circuit"

    return {"records": records, "plan_trace": _trace(state, trace)}


def retrieve_documents(state: AgentState, config: RunnableConfig) -> dict:
    """
    MANDATORY retrieval. The recipe decides the queries; the model has no say.
    On a retry pass (driven by the evidence gate) broadened queries are used.
    """
    intent = Intent(state.get("intent") or Intent.GENERAL.value)
    records = state.get("records") or {}
    account = records.get("account") or {}
    account_name = account.get("name") or ""
    raw_query = state.get("original_query") or ""
    attempt = state.get("retrieval_attempt", 0)

    if attempt == 0:
        queries = retrieval_queries(intent, account_name, raw_query)
    else:
        queries = broadened_queries(intent, raw_query)

    citations = list(state.get("citations") or [])
    seen = {(c.get("source_id"), c.get("page"), c.get("excerpt", "")[:60]) for c in citations}

    for q in queries:
        if not q:
            continue
        results = search_documents.invoke({"query": q, "include_deprecated": False}, config=config)
        if not isinstance(results, list):
            continue
        for r in results:
            if not isinstance(r, dict) or "source_id" not in r:
                continue
            key = (r.get("source_id"), r.get("page"), r.get("excerpt", "")[:60])
            if key in seen:
                continue
            seen.add(key)
            citations.append(r)

    # Historical ticket resolutions enter as authority-15 citations so the
    # resolver can deterministically place them in overruled_sources.
    for h in records.get("history", []):
        key = (f"HIST-{h.get('ticket_id')}", 0, (h.get("resolution") or "")[:60])
        if key in seen:
            continue
        seen.add(key)
        citations.append({
            "source_id": f"HIST-{h.get('ticket_id')}",
            "title": f"Historical Ticket {h.get('ticket_id')} Resolution",
            "status": "historical",
            "authority": 15,
            "scope": h.get("account_id", "global"),
            "page": 0,
            "excerpt": h.get("resolution", ""),
        })

    doc_count = sum(1 for c in citations if (c.get("authority") or 0) >= 20)
    return {
        "citations": citations,
        "retrieval_attempt": attempt + 1,
        "plan_trace": _trace(state, f"retrieval pass {attempt + 1}: {len(queries)} queries, {doc_count} doc chunks"),
    }


def extract_capabilities(state: AgentState) -> dict:
    """
    Derive the subject account's contract terms FROM THE RETRIEVED CHUNKS.
    Nothing is hard-coded per account: if the account has no contract in the
    corpus, capabilities stay at neutral defaults and standard policy applies.
    """
    records = state.get("records") or {}
    subject = records.get("subject_account_id") or ""
    citations = state.get("citations") or []

    contract_chunks = [
        c for c in citations
        if c.get("scope") and c.get("scope") != "global" and c.get("scope") == subject
    ]

    if not contract_chunks:
        # No agreement in scope. Every flag stays neutral and every numeric term
        # stays null, so nothing can be mistaken for an extracted finding.
        return {
            "capabilities": sanitize_capabilities({"contract_found": False}),
            "plan_trace": _trace(state, "no account-scoped contract retrieved; standard policy applies"),
        }

    contract_text = "\n\n".join(c.get("excerpt", "") for c in contract_chunks)

    caps = extract_capabilities_regex(contract_text)

    # Let the LLM refine the structured extraction; regex result is the fallback.
    try:
        resp = plain_llm.invoke([
            SystemMessage(content=CAPABILITY_EXTRACTION_PROMPT + contract_text)
        ])
        parsed = _parse_json_loose(resp.content)
        if parsed:
            merged = dict(caps)
            for k in CAPABILITY_DEFAULTS:
                if k in ("contract_found",):
                    continue
                if k in parsed and parsed[k] is not None:
                    merged[k] = parsed[k]
            merged["contract_found"] = True
            # CRITICAL: if EITHER the regex OR the LLM found "defers to standard",
            # then no waiver is possible — the regex match is authoritative.
            if caps.get("cancellation_defers_to_standard") or merged.get("cancellation_defers_to_standard"):
                merged["cancellation_defers_to_standard"] = True
                merged["has_cancellation_waiver"] = False
            caps = merged
    except Exception:
        pass

    caps = sanitize_capabilities(caps)

    return {
        "capabilities": caps,
        "plan_trace": _trace(
            state,
            f"contract terms from {len(contract_chunks)} chunk(s): "
            f"waiver={caps.get('has_cancellation_waiver')}, "
            f"defers_to_standard={caps.get('cancellation_defers_to_standard')}, "
            f"custom_credit={caps.get('has_custom_credit_rule')}, "
            f"custom_sla_p1_min={caps.get('custom_sla_p1_minutes')}",
        ),
    }


def compute(state: AgentState, config: RunnableConfig) -> dict:
    """
    MANDATORY calculation. Parameters come from DB rows and extracted contract
    terms — never from the model's arithmetic.

    Phase 1 fix: loops over ALL resolved entities instead of taking only [0].
    Phase 3: uses policy_params extracted from retrieved documents.
    """
    intent = Intent(state.get("intent") or Intent.GENERAL.value)
    records = state.get("records") or {}
    caps = state.get("capabilities") or dict(CAPABILITY_DEFAULTS)
    citations = state.get("citations") or []

    # Phase 3: extract policy parameters from the global-scope chunks
    policy = extract_policy_params(citations)

    orders = [r for r in (records.get("orders") or {}).values() if _ok(r)]
    tickets = [r for r in (records.get("tickets") or {}).values() if _ok(r)]

    computed: dict = {}

    if intent is Intent.CANCELLATION and orders:
        # Multi-entity: compute for each order
        results = []
        for order in orders:
            booked = order.get("booked_at")
            requested = order.get("cancellation_requested_at")

            if booked and requested:
                elapsed = minutes_between.invoke(
                    {"start_iso": booked, "end_iso": requested}, config=config
                )
                basis = f"cancellation_requested_at ({requested}) minus booked_at ({booked})"
            elif booked:
                elapsed = minutes_since_booking.invoke({"booked_at_iso": booked}, config=config)
                basis = f"snapshot minus booked_at ({booked})"
            else:
                elapsed, basis = -1.0, "booked_at missing"

            has_waiver = bool(caps.get("has_cancellation_waiver"))
            if caps.get("cancellation_defers_to_standard"):
                has_waiver = False

            fee = cancellation_fee.invoke(
                {
                    "minutes_since_booking": elapsed,
                    "has_custom_waiver": has_waiver,
                    "status": order.get("status", "UNKNOWN"),
                },
                config=config,
            )
            results.append({
                "order_id": order.get("order_id"),
                "elapsed_minutes": elapsed,
                "elapsed_basis": basis,
                "order_status": order.get("status"),
                "contract_waiver_applied": has_waiver,
                "contract_defers_to_standard": bool(caps.get("cancellation_defers_to_standard")),
                "result": fee,
            })

        computed = {
            "kind": "cancellation",
            "results": results,
            # For backward compat / single-order answers
            "order_id": results[0]["order_id"] if results else None,
            "elapsed_minutes": results[0]["elapsed_minutes"] if results else None,
            "order_status": results[0]["order_status"] if results else None,
            "contract_waiver_applied": results[0]["contract_waiver_applied"] if results else None,
            "contract_defers_to_standard": results[0]["contract_defers_to_standard"] if results else None,
            "result": results[0]["result"] if results else {},
            "policy_params": {
                "free_window_minutes": policy.free_window_minutes,
                "fee_after_window_inr": policy.fee_after_window_inr,
                "fee_after_pickup_inr": policy.fee_after_pickup_inr,
                "source": policy.fee_after_window_source,
            },
        }

    elif intent is Intent.SERVICE_CREDIT and orders:
        results = []
        for order in orders:
            window_end = order.get("pickup_window_end")
            if window_end:
                delay = pickup_delay_hours.invoke({"pickup_window_end_iso": window_end}, config=config)
                basis = f"snapshot minus pickup_window_end ({window_end})"
            else:
                delay = -1.0
                basis = "pickup_window_end missing on record"

            has_custom = bool(caps.get("has_custom_credit_rule"))
            credit = service_credit.invoke(
                {
                    "delay_hours": delay,
                    "carrier_fault": bool(order.get("carrier_fault")),
                    "shipment_fee_inr": float(order.get("amount_inr") or 0.0),
                    "has_custom_credit_rule": has_custom,
                    "custom_threshold_hours": float(caps.get("credit_threshold_hours") or 0.0),
                    "custom_fixed_credit_inr": float(caps.get("credit_fixed_inr") or 0.0),
                },
                config=config,
            )
            standard = service_credit.invoke(
                {
                    "delay_hours": delay,
                    "carrier_fault": bool(order.get("carrier_fault")),
                    "shipment_fee_inr": float(order.get("amount_inr") or 0.0),
                    "has_custom_credit_rule": False,
                },
                config=config,
            )
            results.append({
                "order_id": order.get("order_id"),
                "delay_hours": delay,
                "delay_basis": basis,
                "carrier_fault": bool(order.get("carrier_fault")),
                "shipment_fee_inr": order.get("amount_inr"),
                "contract_rule_applied": has_custom,
                "result": credit,
                "standard_policy_result": standard,
            })

        computed = {
            "kind": "service_credit",
            "results": results,
            "order_id": results[0]["order_id"] if results else None,
            "delay_hours": results[0]["delay_hours"] if results else None,
            "result": results[0]["result"] if results else {},
            "standard_policy_result": results[0]["standard_policy_result"] if results else {},
            "contract_rule_applied": results[0]["contract_rule_applied"] if results else False,
            "override_visible": (
                results[0]["contract_rule_applied"]
                and results[0]["standard_policy_result"].get("credit_inr") != results[0]["result"].get("credit_inr")
            ) if results else False,
            "policy_params": {
                "threshold_hours": policy.credit_delay_threshold_hours,
                "cap_inr": policy.credit_cap_inr,
                "pct_of_fee": policy.credit_pct_of_fee,
                "source": policy.credit_threshold_source,
            },
        }

    elif intent is Intent.SLA and tickets:
        results = []
        account = records.get("account") or {}
        for ticket in tickets:
            priority = (ticket.get("priority") or "p2").lower()
            custom_p1 = caps.get("custom_sla_p1_minutes")

            target, source = get_sla_target_minutes(
                policy, account.get("plan", "Standard"), priority,
                custom_p1_minutes=float(custom_p1) if custom_p1 else None,
            )

            remaining = sla_remaining_minutes.invoke(
                {"created_at_iso": ticket.get("created_at"), "sla_minutes": target}, config=config
            )
            elapsed = round(target - remaining, 2)
            results.append({
                "ticket_id": ticket.get("ticket_id"),
                "priority": priority,
                "created_at": ticket.get("created_at"),
                "target_minutes": target,
                "target_source": source,
                "elapsed_minutes": elapsed,
                "remaining_minutes": remaining,
                "breached": remaining < 0,
            })

        computed = {
            "kind": "sla",
            "results": results,
            "ticket_id": results[0]["ticket_id"] if results else None,
            "priority": results[0]["priority"] if results else None,
            "target_minutes": results[0]["target_minutes"] if results else None,
            "target_source": results[0]["target_source"] if results else None,
            "elapsed_minutes": results[0]["elapsed_minutes"] if results else None,
            "remaining_minutes": results[0]["remaining_minutes"] if results else None,
            "breached": results[0]["breached"] if results else False,
        }

    elif intent is Intent.SECURITY:
        computed = {
            "kind": "security",
            "ticket_ids": [t.get("ticket_id") for t in tickets],
            "requires_escalation": True,
        }

    elif intent is Intent.ORDER_STATUS:
        computed = {
            "kind": "order_status",
            "orders": [
                {"order_id": o.get("order_id"), "status": o.get("status"), "carrier": o.get("carrier")}
                for o in orders
            ],
            "denied_lookups": [
                oid for oid, rec in (records.get("orders") or {}).items() if not _ok(rec)
            ],
        }

    elif intent is Intent.PRODUCT_ISSUE:
        # Product issues don't have a numeric calculation but benefit from
        # having the order/ticket context formatted for the composer
        computed = {
            "kind": "product_issue",
            "ticket_ids": [t.get("ticket_id") for t in tickets],
            "order_ids": [o.get("order_id") for o in orders],
            "orders_summary": [
                {"order_id": o.get("order_id"), "status": o.get("status"), "carrier": o.get("carrier")}
                for o in orders
            ],
        }

    return {
        "computed": computed,
        "plan_trace": _trace(state, f"computed: {computed.get('kind', 'none')}, entities={len(orders)+len(tickets)}"),
    }


def resolve_conflicts(state: AgentState) -> dict:
    """
    Rank sources. Uses citations already collected by the recipe path; on the
    ReAct path it scrapes them out of the current turn's tool messages.
    """
    citations = list(state.get("citations") or [])
    pending_draft = state.get("pending_draft")

    if not citations:
        import ast
        messages = state.get("messages", [])
        last_human_idx = 0
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage):
                last_human_idx = i
        for msg in messages[last_human_idx:]:
            if not isinstance(msg, ToolMessage):
                continue
            try:
                try:
                    data = json.loads(msg.content)
                except (json.JSONDecodeError, TypeError):
                    data = ast.literal_eval(msg.content)
            except Exception:
                continue
            if isinstance(data, list) and data and isinstance(data[0], dict) and "source_id" in data[0]:
                citations.extend(data)
            elif isinstance(data, dict):
                if "draft_id" in data and data.get("status") == "draft":
                    pending_draft = data
                if data.get("resolution") and data.get("ticket_id"):
                    citations.append({
                        "source_id": f"HIST-{data['ticket_id']}",
                        "title": f"Historical Ticket {data['ticket_id']} Resolution",
                        "status": "historical",
                        "authority": 15,
                        "scope": data.get("account_id", "global"),
                        "page": 0,
                        "excerpt": data["resolution"],
                    })

    records = state.get("records") or {}
    subject = records.get("subject_account_id")
    if not subject:
        ctx = _ctx_of(state)
        subject = (ctx.get("account_id") if isinstance(ctx, dict) else getattr(ctx, "account_id", None)) or ""

    resolution = resolve_and_score(citations, subject, state.get("original_query") or "")

    return {"citations": citations, "resolution": resolution, "pending_draft": pending_draft}


def evidence_gate(state: AgentState) -> dict:
    """
    Refuse to answer a policy-bearing question with zero authoritative sources.
    This is the guard that turns the earlier silent failure into either a retry
    or an honest, escalation-offering refusal.

    Phase 1 fix: also runs on the ReAct path, not just the recipe path.
    """
    intent_str = state.get("intent")
    # On the ReAct path, intent may not be set — classify now from the query
    if not intent_str:
        from app.agent.planner import classify_intent as _classify
        intent = _classify(state.get("original_query") or "")
    else:
        intent = Intent(intent_str)

    citations = state.get("citations") or []
    doc_citations = [c for c in citations if (c.get("authority") or 0) >= 20]

    if intent in POLICY_BEARING and not doc_citations:
        resolution = dict(state.get("resolution") or {})
        resolution.update({
            "confidence": "low",
            "needs_human": True,
            "explanation": (
                "No authoritative policy or contract text could be retrieved for this question. "
                "Refusing to state a fee, credit or SLA outcome without a source. Escalate for human review."
            ),
            "evidence_gate": "failed",
        })
        return {
            "resolution": resolution,
            "plan_trace": _trace(state, "evidence_gate FAILED: no authoritative sources"),
        }

    return {"plan_trace": _trace(state, f"evidence_gate ok: {len(doc_citations)} authoritative chunks")}


def maybe_prepare_action(state: AgentState, config: RunnableConfig) -> dict:
    """
    Draft an escalation where policy clearly requires one: security incidents,
    and breached SLAs. Still a DRAFT — the HITL interrupt gates execution.
    """
    intent = Intent(state.get("intent") or Intent.GENERAL.value)
    records = state.get("records") or {}
    computed = state.get("computed") or {}
    subject = records.get("subject_account_id")

    tickets = [r for r in (records.get("tickets") or {}).values() if _ok(r)]
    ticket_id = tickets[0].get("ticket_id") if tickets else None

    payload = None
    if intent is Intent.SECURITY and ticket_id:
        payload = {
            "ticket_id": ticket_id,
            "account_id": subject,
            "priority": "critical",
            "reason": "Security incident: possible production API key exposure. Immediate credential rotation and investigation required per current Support Policy.",
        }
    elif intent is Intent.SLA and computed.get("breached"):
        payload = {
            "ticket_id": computed.get("ticket_id"),
            "account_id": subject,
            "priority": "high",
            "reason": (
                f"SLA breached on {computed.get('ticket_id')}: "
                f"{computed.get('elapsed_minutes')} min elapsed against a "
                f"{computed.get('target_minutes')} min target ({computed.get('target_source')})."
            ),
        }

    if not payload:
        return {}

    draft = prepare_action.invoke(
        {"action_type": "create_escalation", "payload": payload}, config=config
    )
    if isinstance(draft, dict) and draft.get("draft_id"):
        return {
            "pending_draft": draft,
            "plan_trace": _trace(state, f"prepared escalation draft {draft['draft_id']}"),
        }
    return {"plan_trace": _trace(state, f"escalation draft not created: {draft}")}


# ==========================================================================
# REACT FALLBACK PATH
# ==========================================================================
def call_model(state: AgentState, config: RunnableConfig) -> dict:
    messages = state.get("messages", [])
    ctx = _ctx_of(state)
    role = ctx.get("role", "unknown") if isinstance(ctx, dict) else getattr(ctx, "role", "unknown")
    acct = ctx.get("account_id", None) if isinstance(ctx, dict) else getattr(ctx, "account_id", None)

    # Pre-fetch: if the model hasn't called any tools yet and this looks like a
    # data-listing query, inject the relevant data as a system message so the model
    # doesn't need to figure out which tool to call.
    query = state.get("original_query") or ""
    iterations = state.get("iterations", 0)

    if iterations == 0:
        prefetch_data = _prefetch_for_general(query, config, ctx)
        if prefetch_data:
            prefetch_msg = SystemMessage(content=f"\nPRE-FETCHED DATA FOR THIS QUERY:\n{prefetch_data}\n\nUse this data to answer. Do not call tools to re-fetch it.")
            messages = list(messages) + [prefetch_msg]

    sys_msg = SystemMessage(
        content=REACT_SYSTEM_PROMPT
        + f"\n\nCURRENT USER CONTEXT:\n- Role: {role}\n- Account ID: {acct}\n"
        "(Already known. Never ask the user for their account ID.)"
    )
    invoke_msgs = [sys_msg] + [m for m in messages if not isinstance(m, SystemMessage)]
    response = llm.invoke(invoke_msgs)
    return {"messages": [response], "iterations": state.get("iterations", 0) + 1}


def _prefetch_for_general(query: str, config: RunnableConfig, ctx) -> str:
    """
    For common list/aggregate queries, pre-fetch the data so the model doesn't
    have to figure out which tool to call (it often doesn't).
    """
    q = (query or "").lower()

    # Ticket listing queries
    if any(p in q for p in ["my tickets", "my open tickets", "show me tickets",
                            "list tickets", "all tickets", "open tickets"]):
        results = query_tickets.invoke({"status": "open"}, config=config)
        if isinstance(results, list) and results and not results[0].get("error"):
            formatted = json.dumps(results, indent=2, default=str)
            return f"Open tickets for this user:\n{formatted}"

    # Order listing queries
    if any(p in q for p in ["my orders", "show me orders", "list orders", "all orders"]):
        results = query_orders.invoke({}, config=config)
        if isinstance(results, list) and results and not results[0].get("error"):
            formatted = json.dumps(results, indent=2, default=str)
            return f"Orders for this user:\n{formatted}"

    # SLA / breach aggregate queries
    if any(p in q for p in ["near sla", "sla breach", "near breach", "at risk",
                            "which accounts"]):
        results = detect_issues.invoke({}, config=config)
        if isinstance(results, list) and results and not results[0].get("error"):
            formatted = json.dumps(results, indent=2, default=str)
            return f"Detected issues across all accounts:\n{formatted}"

    return ""


def should_continue(state: AgentState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return "resolve"
    if state.get("iterations", 0) >= 8:
        return "resolve"
    last = messages[-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "resolve"


# ==========================================================================
# HITL + compose
# ==========================================================================
def human_interrupt(state: AgentState) -> dict:
    draft = state.get("pending_draft")
    if draft:
        res = interrupt({"action_required": True, "draft": draft})
        return {"user_confirmation": res}
    return {}


def execute_or_skip(state: AgentState, config: RunnableConfig) -> dict:
    conf = state.get("user_confirmation")
    draft = state.get("pending_draft")
    if not (draft and conf):
        return {}

    decision = conf if isinstance(conf, bool) else (conf.get("confirm") if isinstance(conf, dict) else False)
    ctx = _ctx_of(state)

    if decision:
        res = _execute_action(draft["draft_id"], ctx)
        return {"execute_result": res, "pending_draft": None}

    from app.tools.actions import cancel_action
    cancel_action(draft["draft_id"], ctx)
    return {
        "execute_result": {"status": "cancelled", "message": "User cancelled the action."},
        "pending_draft": None,
    }


def compose_answer(state: AgentState, config: RunnableConfig) -> dict:
    """Narrate an answer over evidence and numbers that are already fixed."""
    resolution = state.get("resolution") or {}
    computed = state.get("computed") or {}
    records = state.get("records") or {}
    caps = state.get("capabilities") or {}
    exec_res = state.get("execute_result")
    intent_str = state.get("intent")

    # ---- Access-denied path: answer without any cross-account detail ----
    if records.get("access_blocked"):
        try:
            response = plain_llm.invoke([
                SystemMessage(content=DENIAL_PROMPT),
                SystemMessage(content=f"USER QUESTION: {state.get('original_query', '')}"),
            ])
            answer = (response.content or "").strip()
            if answer and len(answer) > 10:
                structured = parse_answer(answer)
                return {"final_answer": structured.model_dump_json()}
        except Exception:
            pass
        # Fallback
        denial = StructuredAnswer(
            verdict="That record could not be found. I can only access records on your own account.",
            reasoning="The requested record is not available from your account. I can help with your own orders and tickets.",
            confidence="high",
            suggested_action="Ask about your own orders or tickets instead.",
        )
        return {"final_answer": denial.model_dump_json()}

    # ---- ReAct path: the model already composed its answer in the message stream.
    # Do NOT overwrite it with a new compose call. Extract the last AI message.
    if not intent_str or intent_str == Intent.GENERAL.value:
        messages = state.get("messages", [])
        # Find the last AI message that has content (the model's final answer)
        last_ai_content = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content and not isinstance(msg, (HumanMessage, ToolMessage, SystemMessage)):
                if not getattr(msg, "tool_calls", None):  # Skip messages that are just tool calls
                    last_ai_content = msg.content
                    break

        if last_ai_content and len(last_ai_content) > 20:
            # The model already answered. Package it as a structured answer.
            structured = StructuredAnswer(
                verdict=last_ai_content[:500] if len(last_ai_content) > 500 else last_ai_content,
                reasoning="",
                confidence="medium",
            )
            return {"final_answer": structured.model_dump_json()}

    # ---- Recipe path: compose from computed data ----
    slim_records = {
        "account_name": (records.get("account") or {}).get("name"),
        "account_plan": (records.get("account") or {}).get("plan"),
        "orders": records.get("orders") or {},
        "tickets": records.get("tickets") or {},
        "relevant_past_ticket_advice": [
            {"ticket_id": h.get("ticket_id"), "what_the_agent_said": h.get("resolution")}
            for h in (records.get("history") or [])
        ],
        "records_not_accessible": bool(records.get("denied")),
    }

    contract_summary = {
        "contract_on_file": bool(caps.get("contract_found")),
        "waives_cancellation_fee": caps.get("has_cancellation_waiver"),
        "defers_to_standard_cancellation_policy": caps.get("cancellation_defers_to_standard"),
        "has_own_service_credit_rule": caps.get("has_custom_credit_rule"),
        "own_credit_threshold_hours": caps.get("credit_threshold_hours"),
        "own_credit_amount_inr": caps.get("credit_fixed_inr"),
        "own_p1_response_target_minutes": caps.get("custom_sla_p1_minutes"),
        "clause_note": caps.get("notes"),
    }

    binding = resolution.get("binding_source") or {}
    supporting = resolution.get("supporting_sources") or []
    overruled = resolution.get("overruled_sources") or []

    parts = [
        f"USER QUESTION: {state.get('original_query', '')}",
        f"\nRECORDS:\n{json.dumps(slim_records, indent=2, default=str)}",
        f"\nWHAT THE RETRIEVED AGREEMENT SAYS (null means the agreement is silent):\n"
        f"{json.dumps(contract_summary, indent=2, default=str)}",
        f"\nCALCULATED OUTCOME — these figures are final, restate them, never recompute:\n"
        f"{json.dumps(computed, indent=2, default=str)}",
        f"\nGOVERNING SOURCE:\n{json.dumps(binding, indent=2, default=str)}",
        f"\nSUPPORTING SOURCES (same current policy set — read these for specific "
        f"amounts and thresholds the governing source refers to):\n"
        f"{json.dumps(supporting[:4], indent=2, default=str)}",
        f"\nSUPERSEDED OR NON-BINDING SOURCES (deprecated policy, past agent advice):\n"
        f"{json.dumps(overruled[:4], indent=2, default=str)}",
        f"\nRESOLVER: confidence={resolution.get('confidence')}, "
        f"needs_human={resolution.get('needs_human')}, why={resolution.get('explanation')}",
    ]
    if exec_res:
        parts.append(f"\nACTION RESULT:\n{json.dumps(exec_res, indent=2, default=str)}")
    if state.get("pending_draft"):
        parts.append(
            f"\nA DRAFT ACTION IS AWAITING USER CONFIRMATION:\n"
            f"{json.dumps(state['pending_draft'], indent=2, default=str)}"
        )
    if resolution.get("evidence_gate") == "failed":
        parts.append(
            "\nIMPORTANT: no authoritative policy text was retrieved. Do NOT state any fee, "
            "credit or SLA figure. Say the governing policy could not be confirmed and offer escalation."
        )
    if slim_records["records_not_accessible"]:
        parts.append(
            "\nNOTE: one or more records named in the question are not accessible to this user. "
            "Answer only from what IS available. Do not name, describe or speculate about the "
            "inaccessible records beyond saying they were not found."
        )

    response = plain_llm.invoke([SystemMessage(content=COMPOSE_PROMPT + "\n\n" + "\n".join(parts))])
    answer = (response.content or "").strip()

    # Parse into structured format — never send raw JSON to the frontend
    if answer and len(answer) > 10:
        structured = parse_answer(answer)
    else:
        structured = _fallback_structured(state)

    return {"final_answer": structured.model_dump_json()}


def _fallback_structured(state: AgentState) -> StructuredAnswer:
    """Build a structured answer from computed data when the LLM compose fails."""
    computed = state.get("computed") or {}
    resolution = state.get("resolution") or {}
    records = state.get("records") or {}
    binding = (resolution.get("binding_source") or {})

    if records.get("access_blocked"):
        return StructuredAnswer(
            verdict="That record could not be found. I can only access records on your own account.",
            reasoning="The requested record is not available from your account. I can help with your own orders and tickets.",
            confidence="high",
            suggested_action="Ask about your own orders or tickets instead.",
        )

    kind = computed.get("kind")

    if kind == "cancellation":
        results = computed.get("results") or [computed]
        if len(results) == 1:
            r = results[0]
            fee = (r.get("result") or {}).get("fee_inr", "unknown")
            reason = (r.get("result") or {}).get("reason", "")
            return StructuredAnswer(
                verdict=f"Cancellation fee: INR {fee}. {reason}",
                reasoning=reason,
                confidence=resolution.get("confidence", "medium"),
                citations=[{"title": binding.get("title", ""), "excerpt": "", "authority": binding.get("authority", 0), "status": "current"}] if binding.get("title") else [],
            )
        else:
            lines = [f"{r['order_id']}: INR {(r.get('result') or {}).get('fee_inr', '?')} — {(r.get('result') or {}).get('reason', '')}" for r in results]
            return StructuredAnswer(
                verdict="Cancellation fees: " + "; ".join(lines),
                reasoning="See per-order breakdown above.",
                confidence=resolution.get("confidence", "medium"),
            )

    elif kind == "service_credit":
        results = computed.get("results") or [computed]
        r = results[0]
        credit = (r.get("result") or {}).get("credit_inr", 0)
        eligible = (r.get("result") or {}).get("eligible", False)
        reason = (r.get("result") or {}).get("reason", "")
        return StructuredAnswer(
            verdict=f"{'Eligible' if eligible else 'Not eligible'} for a service credit of INR {credit}. {reason}",
            reasoning=reason,
            confidence=resolution.get("confidence", "medium"),
            citations=[{"title": binding.get("title", ""), "excerpt": "", "authority": binding.get("authority", 0), "status": "current"}] if binding.get("title") else [],
        )

    elif kind == "sla":
        breached = computed.get("breached", False)
        remaining = computed.get("remaining_minutes", 0)
        target = computed.get("target_minutes", 0)
        return StructuredAnswer(
            verdict=f"SLA {'BREACHED' if breached else 'within target'}. Target: {target} minutes. Remaining: {remaining} minutes.",
            reasoning=f"Elapsed: {computed.get('elapsed_minutes', 0)} min vs target {target} min ({computed.get('target_source', 'standard')}).",
            confidence="high",
            suggested_action="Escalate immediately." if breached else None,
        )

    return StructuredAnswer(
        verdict="Analysis complete. See the tool results above for details.",
        reasoning=resolution.get("explanation", ""),
        confidence=resolution.get("confidence", "medium"),
    )


# ==========================================================================
# Routing
# ==========================================================================
def route_after_classify(state: AgentState) -> str:
    intent = Intent(state.get("intent") or Intent.GENERAL.value)
    return "agent" if intent is Intent.GENERAL else "gather_entities"


def route_after_entities(state: AgentState) -> str:
    """
    Short circuit straight to the answer when every requested record was refused.
    Retrieving policy text and running the resolver over a question we are not
    allowed to answer produces confusing low-confidence noise and risks leaking
    cross-account identifiers into the narration.
    """
    records = state.get("records") or {}
    if records.get("access_blocked"):
        return "compose_answer"
    return "retrieve_documents"


def route_after_gate(state: AgentState) -> str:
    intent = Intent(state.get("intent") or Intent.GENERAL.value)
    resolution = state.get("resolution") or {}
    citations = state.get("citations") or []
    doc_citations = [c for c in citations if (c.get("authority") or 0) >= 20]

    # One retry with broadened queries before giving up
    if (
        intent in POLICY_BEARING
        and not doc_citations
        and state.get("retrieval_attempt", 0) < 2
    ):
        return "retrieve_documents"

    computed = state.get("computed") or {}
    if intent is Intent.SECURITY or (intent is Intent.SLA and computed.get("breached")):
        if resolution.get("evidence_gate") != "failed":
            return "maybe_prepare_action"

    return "compose_answer"


def route_after_action(state: AgentState) -> str:
    return "human_interrupt" if state.get("pending_draft") else "compose_answer"


def route_after_resolve_react(state: AgentState) -> str:
    if state.get("pending_draft"):
        return "human_interrupt"
    return "compose_answer"


# ==========================================================================
# Graph
# ==========================================================================
builder = StateGraph(AgentState)

# recipe path
builder.add_node("classify", classify)
builder.add_node("gather_entities", gather_entities)
builder.add_node("retrieve_documents", retrieve_documents)
builder.add_node("extract_capabilities", extract_capabilities)
builder.add_node("compute", compute)
builder.add_node("resolve", resolve_conflicts)
builder.add_node("evidence_gate", evidence_gate)
builder.add_node("maybe_prepare_action", maybe_prepare_action)

# react fallback
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.add_node("resolve_react", resolve_conflicts)
builder.add_node("evidence_gate_react", evidence_gate)

# shared tail
builder.add_node("human_interrupt", human_interrupt)
builder.add_node("execute_or_skip", execute_or_skip)
builder.add_node("compose_answer", compose_answer)

builder.add_edge(START, "classify")
builder.add_conditional_edges(
    "classify", route_after_classify,
    {"agent": "agent", "gather_entities": "gather_entities"},
)

builder.add_conditional_edges(
    "gather_entities", route_after_entities,
    {"retrieve_documents": "retrieve_documents", "compose_answer": "compose_answer"},
)
builder.add_edge("retrieve_documents", "extract_capabilities")
builder.add_edge("extract_capabilities", "compute")
builder.add_edge("compute", "resolve")
builder.add_edge("resolve", "evidence_gate")
builder.add_conditional_edges(
    "evidence_gate", route_after_gate,
    {
        "retrieve_documents": "retrieve_documents",
        "maybe_prepare_action": "maybe_prepare_action",
        "compose_answer": "compose_answer",
    },
)
builder.add_conditional_edges(
    "maybe_prepare_action", route_after_action,
    {"human_interrupt": "human_interrupt", "compose_answer": "compose_answer"},
)

builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "resolve": "resolve_react"})
builder.add_edge("tools", "agent")
builder.add_edge("resolve_react", "evidence_gate_react")
builder.add_conditional_edges(
    "evidence_gate_react", route_after_resolve_react,
    {"human_interrupt": "human_interrupt", "compose_answer": "compose_answer"},
)

builder.add_edge("human_interrupt", "execute_or_skip")
builder.add_edge("execute_or_skip", "compose_answer")
builder.add_edge("compose_answer", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
