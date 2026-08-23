from app.auth import UserContext
from typing import Optional


class ForbiddenError(Exception):
    def __init__(self, message="Forbidden"):
        super().__init__(message)


def scoped_account_ids(ctx) -> Optional[list[str]]:
    """Return list of account IDs this user may access, or None for unrestricted."""
    if ctx is None:
        return None
    if isinstance(ctx, dict):
        role = ctx.get("role", "")
        acct = ctx.get("account_id")
    else:
        role = getattr(ctx, "role", "")
        acct = getattr(ctx, "account_id", None)

    if role == "customer" and acct:
        return [acct]
    return None  # internal sees all


def enforce(account_id: str, ctx) -> None:
    """Raises ForbiddenError if ctx can't access account_id."""
    allowed = scoped_account_ids(ctx)
    if allowed is not None and account_id not in allowed:
        raise ForbiddenError()  # no existence leak


def get_account_id_from_ctx(ctx) -> Optional[str]:
    """Extract account_id from context (dict or object)."""
    if ctx is None:
        return None
    if isinstance(ctx, dict):
        return ctx.get("account_id")
    return getattr(ctx, "account_id", None)


def get_role_from_ctx(ctx) -> str:
    if ctx is None:
        return "unknown"
    if isinstance(ctx, dict):
        return ctx.get("role", "unknown")
    return getattr(ctx, "role", "unknown")
