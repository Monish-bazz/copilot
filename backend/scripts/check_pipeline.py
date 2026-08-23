"""
Pre-flight diagnostic. Run this before testing the chat UI.

    cd backend
    python -m scripts.check_pipeline

Verifies, in order:
  1. env + DB connectivity
  2. SQLite/Postgres rows are ingested and the new columns are populated
  3. the pgvector collection actually returns chunks (the silent blocker)
  4. ACL denies cross-account reads
  5. the planner routes all eight demo prompts correctly

Exits non-zero if anything that would break the demo is wrong.
"""
from __future__ import annotations

import sys

FAIL: list[str] = []
WARN: list[str] = []


def ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def bad(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    FAIL.append(msg)


def warn(msg: str) -> None:
    print(f"  [warn] {msg}")
    WARN.append(msg)


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


# ---------------------------------------------------------------- 1. config
section("1. Configuration")
try:
    from app.config import DATABASE_URL, NVIDIA_API_KEY, NIM_MODEL, SNAPSHOT_TS

    ok(f"snapshot = {SNAPSHOT_TS.isoformat()}")
    if DATABASE_URL:
        host = DATABASE_URL.split("@")[-1].split("/")[0] if "@" in DATABASE_URL else "configured"
        ok(f"DATABASE_URL set (host: {host})")
    else:
        bad("DATABASE_URL / supabase not set — every tool will fail")
    if NVIDIA_API_KEY:
        ok("NVIDIA API key set")
    else:
        bad("NVIDIA API key not set")
    ok(f"model = {NIM_MODEL}")
except Exception as e:
    bad(f"cannot import config: {e}")
    sys.exit(1)


# ------------------------------------------------------------ 2. structured
section("2. Structured data")
try:
    from sqlmodel import Session, select
    from app.db.engine import engine
    from app.db.models import Account, Order, Ticket

    if engine is None:
        bad("engine is None — DATABASE_URL missing")
    else:
        with Session(engine) as s:
            accounts = s.exec(select(Account)).all()
            orders = s.exec(select(Order)).all()
            tickets = s.exec(select(Ticket)).all()

        print(f"  accounts={len(accounts)} orders={len(orders)} tickets={len(tickets)}")
        if len(accounts) >= 4:
            ok("accounts ingested")
        else:
            bad(f"expected 4 accounts, found {len(accounts)} — run: python -m app.db.ingest_excel")
        if len(orders) >= 6:
            ok("orders ingested")
        else:
            bad(f"expected 6 orders, found {len(orders)} — run: python -m app.db.ingest_excel")
        if len(tickets) >= 7:
            ok("tickets ingested")
        else:
            bad(f"expected 7 tickets, found {len(tickets)}")

        by_id = {o.order_id: o for o in orders}

        # The columns that were missing before and that Case B/C depend on
        o2001 = by_id.get("ORD-2001")
        if o2001 is None:
            bad("ORD-2001 missing")
        elif o2001.cancellation_requested_at is None:
            bad("ORD-2001.cancellation_requested_at is NULL — re-run ingest_excel (Case B will fail)")
        else:
            ok(f"ORD-2001.cancellation_requested_at = {o2001.cancellation_requested_at}")

        o2002 = by_id.get("ORD-2002")
        if o2002 is None:
            bad("ORD-2002 missing")
        else:
            if o2002.pickup_window_end is None:
                bad("ORD-2002.pickup_window_end is NULL — re-run ingest_excel (Case C will fail)")
            else:
                ok(f"ORD-2002.pickup_window_end = {o2002.pickup_window_end}")
            if o2002.carrier_fault is not True:
                bad(f"ORD-2002.carrier_fault = {o2002.carrier_fault}, expected True")
            else:
                ok("ORD-2002.carrier_fault = True")

        hist = [t for t in tickets if t.resolution]
        if any(t.ticket_id == "TKT-450" for t in hist):
            ok("TKT-450 historical resolution present (trap available)")
        else:
            warn("TKT-450 has no resolution text — the Case A trap will not be demonstrable")
except Exception as e:
    bad(f"structured data check failed: {e}")


# ---------------------------------------------------------------- 3. vectors
section("3. Vector store (the usual silent blocker)")
try:
    from app.tools.documents import search_documents
    from app.auth import USERS

    internal = USERS["rohit.ops"]
    probe = search_documents("cancellation fee window", ctx=internal)

    if not probe:
        bad("search_documents returned 0 chunks — run: python -m app.rag.ingest_pdfs")
    else:
        ok(f"retrieval returned {len(probe)} chunks")
        sources = sorted({c.get("source_id") for c in probe})
        print(f"  sources: {sources}")

        contract_probe = search_documents("Northstar agreement cancellation waiver", ctx=internal)
        contract_sources = {c.get("source_id") for c in contract_probe}
        if "CTR-NS" in contract_sources:
            ok("Northstar contract chunks retrievable")
        else:
            warn(f"CTR-NS not in top results for a contract query (got {sorted(contract_sources)})")

        prod_probe = search_documents("known issue bulk upload CSV row limit", ctx=internal)
        if "PROD" in {c.get("source_id") for c in prod_probe}:
            ok("Product Operations Guide chunks retrievable")
        else:
            warn("PROD not retrievable for a known-issue query — Cases D/E may degrade")

        if any(c.get("status") == "deprecated" for c in probe):
            bad("deprecated policy leaked into default results")
        else:
            ok("deprecated policy excluded by default")
except Exception as e:
    bad(f"vector store check failed: {e}")


# -------------------------------------------------------------------- 4. ACL
section("4. Tool-layer ACL")
try:
    from app.auth import USERS
    from app.tools.lookup import get_order, query_tickets

    priya = USERS["priya.northstar"]
    rohit = USERS["rohit.ops"]

    cross = get_order("ORD-2001", priya)
    if cross.get("error") == "not_found":
        ok("customer cross-account order read denied without existence leak")
    else:
        bad(f"ACL leak: priya reading ORD-2001 returned {cross}")

    own = get_order("ORD-1001", priya)
    if own.get("error"):
        bad(f"customer cannot read own order ORD-1001: {own}")
    else:
        ok("customer can read own order")

    internal_read = get_order("ORD-2001", rohit)
    if internal_read.get("error"):
        bad(f"internal agent cannot read ORD-2001: {internal_read}")
    else:
        ok("internal agent can read any order")

    denied = query_tickets(account_id="ACCT-002", ctx=priya)
    if denied and denied[0].get("error") == "access_denied":
        ok("cross-account ticket query returns explicit access_denied")
    else:
        bad(f"cross-account ticket query returned {denied}")
except Exception as e:
    bad(f"ACL check failed: {e}")


# ---------------------------------------------------------------- 5. planner
section("5. Planner routing")
try:
    from app.agent.planner import Intent, classify_intent, extract_entities

    cases = [
        ("A", "Can Northstar cancel order ORD-1001 without a cancellation fee? Explain why based on policies.", Intent.CANCELLATION),
        ("B", "What is the cancellation fee for LumenWorks order ORD-2001? Refer to the specific agreements and policies.", Intent.CANCELLATION),
        ("C", "Is LumenWorks eligible for a service credit on order ORD-2002? If so, what is the credit amount?", Intent.SERVICE_CREDIT),
        ("D", "LumenWorks is getting errors uploading a 4,200-row CSV (TKT-502). Does the Growth plan support this, and what is the workaround?", Intent.PRODUCT_ISSUE),
        ("E", "Northstar order ORD-1001 was collected by the driver 10 minutes ago, but it still shows BOOKED in the portal (TKT-504). Did the pickup fail?", Intent.PRODUCT_ISSUE),
        ("F", "Review open ticket TKT-501. Has the SLA been breached? What are the support targets for Northstar?", Intent.SLA),
        ("G", "What should we do about Axis Labs ticket TKT-505 regarding API key exposure?", Intent.SECURITY),
        ("H", "What is the status of LumenWorks order ORD-2001?", Intent.ORDER_STATUS),
    ]

    for label, prompt, expected in cases:
        got = classify_intent(prompt)
        ents = extract_entities(prompt)
        flat = ents["order_ids"] + ents["ticket_ids"] + ents["account_ids"]
        if got is expected:
            ok(f"Case {label}: {got.value}  entities={flat}")
        else:
            bad(f"Case {label}: expected {expected.value}, got {got.value}")
except Exception as e:
    bad(f"planner check failed: {e}")


# ---------------------------------------------------------------- 6. summary
section("Summary")
if FAIL:
    print(f"  {len(FAIL)} blocking problem(s):")
    for f in FAIL:
        print(f"    - {f}")
if WARN:
    print(f"  {len(WARN)} warning(s):")
    for w in WARN:
        print(f"    - {w}")
if not FAIL and not WARN:
    print("  All checks passed. The demo scenarios should run.")
elif not FAIL:
    print("  No blocking problems. Warnings may degrade specific cases.")

sys.exit(1 if FAIL else 0)
