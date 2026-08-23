from app.rag.ingest_pdfs import get_vector_store
from app.tools.acl import scoped_account_ids


def search_documents(query: str, include_deprecated: bool = False, ctx=None) -> list[dict]:
    """
    Search authoritative documents with scope and status filtering.
    Returns top results with full metadata for the resolver.
    """
    vectorstore = get_vector_store()

    # Fetch more than needed so we can filter client-side
    docs = vectorstore.similarity_search(query=query, k=20)

    allowed_accounts = scoped_account_ids(ctx)

    results = []
    seen_source_ids = set()

    for doc in docs:
        meta = doc.metadata

        # Scope filter: customer only sees global or their own contract docs
        scope = meta.get("scope", "global")
        if allowed_accounts is not None:
            if scope != "global" and scope not in allowed_accounts:
                continue

        # Deprecated filter
        if not include_deprecated and meta.get("status") == "deprecated":
            continue

        source_id = meta.get("source_id", "unknown")

        # Ensure diversity: max 2 chunks per source to avoid one doc dominating
        count_for_source = sum(1 for r in results if r["source_id"] == source_id)
        if count_for_source >= 2:
            continue

        results.append({
            "source_id": source_id,
            "title": meta.get("title", "Unknown"),
            "status": meta.get("status", "unknown"),
            "authority": meta.get("authority", 0),
            "scope": scope,
            "page": meta.get("page_number", 1),
            "excerpt": doc.page_content,
        })

        if len(results) >= 8:
            break

    return results
