"""
Conflict Resolver — deterministic source ranking.

Rules:
1. Customer-scoped current agreement > global current policy (for commercial terms).
2. Current always beats deprecated.
3. Historical tickets (authority 15) never override documents.
4. Product guide is binding for product defects, not for commercial/SLA terms.
5. A genuine unresolved conflict → needs_human. Proximity of authority is NOT a
   conflict: the global policy tier documents are complementary, not competing.
6. Customer retriever drops other accounts' contracts before the LLM sees them.

Sources are split three ways rather than two:
  binding_source     — the one that governs
  supporting_sources — same-tier current documents that fill in detail
                       (e.g. SOP v4 supplying the fee amount that POL v3 refers to)
  overruled_sources  — genuinely superseded or non-authoritative: deprecated
                       policy and historical ticket advice
"""
from typing import Optional

# Documents that make up the single global policy corpus. These are published
# together and are complementary; differing authority between them expresses
# specificity, not disagreement.
GLOBAL_POLICY_TIER = {"POL-V3", "SOP-V4", "PROD"}

COMMERCIAL_KEYWORDS = [
    "cancell", "fee", "refund", "credit", "charge", "waiv", "sla",
    "penalty", "service credit", "billing", "payment", "price",
    "agreement", "contract", "terms",
]

PRODUCT_KEYWORDS = [
    "bug", "known issue", "ki-", "error", "fail", "upload", "csv",
    "webhook", "delay", "http 500", "timeout", "row limit", "api",
    "tracking", "stuck", "portal", "status update",
]


def classify_topic(query: str) -> str:
    """Classify a query as 'commercial', 'product', or 'general'."""
    q = (query or "").lower()
    commercial_score = sum(1 for kw in COMMERCIAL_KEYWORDS if kw in q)
    product_score = sum(1 for kw in PRODUCT_KEYWORDS if kw in q)

    if product_score > commercial_score:
        return "product"
    if commercial_score > 0:
        return "commercial"
    return "general"


def _score(c: dict, topic: str, account_id: str) -> tuple:
    """Higher tuple sorts first."""
    authority = c.get("authority", 0) or 0
    scope = c.get("scope", "global")
    status = c.get("status", "unknown")
    source_id = c.get("source_id", "") or ""

    # Non-authoritative context can never win
    if authority <= 15 or status == "historical":
        return (0, 0, 0, 0)

    status_score = 2 if status == "current" else (1 if status != "deprecated" else 0)

    scope_score = 0
    if scope and scope != "global" and scope == account_id:
        scope_score = 2
    elif scope == "global":
        scope_score = 1

    topic_score = 0
    if topic == "product" and source_id == "PROD":
        topic_score = 3
    elif topic == "commercial" and scope and scope != "global" and scope == account_id:
        topic_score = 3
    elif topic == "commercial" and source_id in ("SOP-V4", "POL-V3"):
        topic_score = 2

    return (status_score, topic_score, scope_score, authority)


def _is_genuine_conflict(a: dict, b: dict, topic: str, account_id: str) -> bool:
    """
    A conflict requires two sources of *equal standing* making competing claims.

    Deliberately strict. Two documents from the global policy tier are
    complementary, so POL-V3 vs SOP-V4 is never a conflict — that false positive
    previously dropped every standard-policy answer to low confidence.
    """
    if a.get("source_id") == b.get("source_id"):
        return False

    a_global_tier = a.get("source_id") in GLOBAL_POLICY_TIER
    b_global_tier = b.get("source_id") in GLOBAL_POLICY_TIER
    if a_global_tier and b_global_tier:
        return False

    # Both non-authoritative — handled elsewhere
    sa = _score(a, topic, account_id)
    sb = _score(b, topic, account_id)
    if sa == (0, 0, 0, 0) or sb == (0, 0, 0, 0):
        return False

    # Genuine ambiguity: identical standing on every dimension, including authority
    return sa == sb


def resolve_and_score(citations: list[dict], account_id: str, query: str = "") -> dict:
    """
    Rank retrieved sources and decide which one governs.

    Args:
        citations: dicts with source_id, title, status, authority, scope, excerpt
        account_id: the SUBJECT account (not necessarily the logged-in user's)
        query: original user query, for topic classification
    """
    if not citations:
        return {
            "binding_source": None,
            "supporting_sources": [],
            "overruled_sources": [],
            "confidence": "low",
            "needs_human": True,
            "explanation": "No relevant documents found.",
            "topic": classify_topic(query),
        }

    # Deduplicate by source_id, keeping the richest excerpt per source
    deduped: dict[str, dict] = {}
    for c in citations:
        sid = c.get("source_id", "unknown")
        if sid not in deduped or len(c.get("excerpt", "")) > len(deduped[sid].get("excerpt", "")):
            deduped[sid] = c
    cits = list(deduped.values())

    topic = classify_topic(query)

    authoritative = [
        c for c in cits
        if (c.get("authority", 0) or 0) > 15 and c.get("status") not in ("deprecated", "historical")
    ]
    deprecated = [c for c in cits if c.get("status") == "deprecated"]
    historical = [
        c for c in cits
        if c.get("status") == "historical" or (c.get("authority", 0) or 0) <= 15
    ]

    if not authoritative:
        # Nothing binding available; report honestly
        fallback = (deprecated + historical)
        return {
            "binding_source": fallback[0] if fallback else None,
            "supporting_sources": [],
            "overruled_sources": fallback[1:] if len(fallback) > 1 else [],
            "confidence": "low",
            "needs_human": True,
            "explanation": (
                "No current authoritative policy or contract was retrieved. "
                "Only deprecated or historical material is available."
            ),
            "topic": topic,
        }

    ranked = sorted(authoritative, key=lambda c: _score(c, topic, account_id), reverse=True)
    binding = ranked[0]
    rest = ranked[1:]

    binding_scope = binding.get("scope")
    binding_is_contract = bool(binding_scope) and binding_scope != "global" and binding_scope == account_id

    # Same-tier current documents support the binding source rather than being overruled by it.
    supporting: list[dict] = []
    overruled: list[dict] = list(deprecated) + list(historical)

    for c in rest:
        c_scope = c.get("scope")
        c_is_other_contract = bool(c_scope) and c_scope != "global" and c_scope != account_id
        if c_is_other_contract:
            # Another account's agreement is irrelevant here (and normally filtered by ACL)
            continue
        if c.get("status") == "current":
            supporting.append(c)
        else:
            overruled.append(c)

    # Confidence
    needs_human = False
    if binding_is_contract:
        confidence = "high"
        explanation = (
            f"'{binding.get('title')}' is the binding source: it is a current agreement "
            f"scoped to this account, so it governs over standard policy."
        )
    elif topic == "product" and binding.get("source_id") == "PROD":
        confidence = "high"
        explanation = "The Product Operations Guide governs product behaviour, limits and known issues."
    else:
        confidence = "high" if supporting else "medium"
        explanation = (
            f"'{binding.get('title')}' governs. No account-specific agreement covers this point, "
            f"so standard policy applies."
        )

    # Genuine conflict check (strict)
    if len(ranked) >= 2 and _is_genuine_conflict(ranked[0], ranked[1], topic, account_id):
        confidence = "low"
        needs_human = True
        explanation = (
            f"Unresolved conflict: '{ranked[0].get('title')}' and '{ranked[1].get('title')}' "
            f"carry equal standing on this question. Human review required."
        )

    return {
        "binding_source": binding,
        "supporting_sources": supporting,
        "overruled_sources": overruled,
        "confidence": confidence,
        "needs_human": needs_human,
        "explanation": explanation,
        "topic": topic,
    }
