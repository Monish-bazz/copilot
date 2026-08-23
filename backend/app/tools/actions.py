import uuid
from sqlmodel import Session
from app.db.engine import engine
from app.db.models import Escalation, FollowupTask, AuditLog
from app.tools.acl import enforce, ForbiddenError, get_role_from_ctx
from datetime import datetime


def prepare_action(action_type: str, payload: dict, ctx) -> dict:
    """
    Prepare a draft action. Does NOT execute — requires explicit confirmation.
    Supported types: create_escalation, create_followup_task
    """
    role = get_role_from_ctx(ctx)
    user_key = ctx.user_key if hasattr(ctx, "user_key") else ctx.get("user_key", "unknown")

    draft_id = str(uuid.uuid4())

    with Session(engine) as session:
        if action_type == "create_escalation":
            if payload.get("account_id"):
                try:
                    enforce(payload["account_id"], ctx)
                except ForbiddenError:
                    return {"error": "forbidden", "message": "Cannot escalate for this account."}

            draft = Escalation(
                draft_id=draft_id,
                order_id=payload.get("order_id"),
                ticket_id=payload.get("ticket_id"),
                account_id=payload.get("account_id"),
                priority=payload.get("priority", "high"),
                reason=payload.get("reason", "No reason provided"),
                status="draft",
                created_by=user_key,
                created_at=datetime.now(),
            )
            session.add(draft)

        elif action_type == "create_followup_task":
            draft = FollowupTask(
                draft_id=draft_id,
                related_id=payload.get("related_id"),
                description=payload.get("description", ""),
                status="draft",
                created_by=user_key,
                created_at=datetime.now(),
            )
            session.add(draft)
        else:
            return {"error": "unknown_action", "message": f"Unknown action type: {action_type}"}

        # Audit
        audit = AuditLog(
            user_key=user_key,
            tool_name="prepare_action",
            arguments_json=f'{{"action_type":"{action_type}","payload":{str(payload)}}}',
            result_summary=f"draft_id={draft_id}",
            timestamp=datetime.now(),
        )
        session.add(audit)
        session.commit()

    return {
        "draft_id": draft_id,
        "action_type": action_type,
        "payload": payload,
        "status": "draft",
        "message": "Action prepared. Awaiting human confirmation.",
    }


def execute_action(draft_id: str, ctx) -> dict:
    """
    Execute a previously prepared draft action.
    Only works if the draft is still in 'draft' status.
    """
    user_key = ctx.user_key if hasattr(ctx, "user_key") else ctx.get("user_key", "unknown")

    with Session(engine) as session:
        # Check escalations
        escalation = session.query(Escalation).filter(Escalation.draft_id == draft_id).first()
        if escalation:
            if escalation.status != "draft":
                return {"error": "invalid_state", "message": f"Action already {escalation.status}."}
            escalation.status = "executed"
            audit = AuditLog(
                user_key=user_key,
                tool_name="execute_action",
                arguments_json=f'{{"draft_id":"{draft_id}"}}',
                result_summary="escalation executed",
                timestamp=datetime.now(),
            )
            session.add(audit)
            session.commit()
            return {"status": "success", "message": "Escalation created and executed successfully."}

        # Check followup tasks
        task = session.query(FollowupTask).filter(FollowupTask.draft_id == draft_id).first()
        if task:
            if task.status != "draft":
                return {"error": "invalid_state", "message": f"Action already {task.status}."}
            task.status = "executed"
            audit = AuditLog(
                user_key=user_key,
                tool_name="execute_action",
                arguments_json=f'{{"draft_id":"{draft_id}"}}',
                result_summary="followup_task executed",
                timestamp=datetime.now(),
            )
            session.add(audit)
            session.commit()
            return {"status": "success", "message": "Follow-up task created and executed successfully."}

        return {"error": "not_found", "message": "Draft not found. Cannot execute."}


def cancel_action(draft_id: str, ctx) -> dict:
    """Cancel a prepared draft action."""
    user_key = ctx.user_key if hasattr(ctx, "user_key") else ctx.get("user_key", "unknown")

    with Session(engine) as session:
        escalation = session.query(Escalation).filter(Escalation.draft_id == draft_id).first()
        if escalation:
            if escalation.status != "draft":
                return {"error": "invalid_state", "message": f"Cannot cancel: already {escalation.status}."}
            escalation.status = "cancelled"
            session.commit()
            return {"status": "cancelled", "message": "Action cancelled."}

        task = session.query(FollowupTask).filter(FollowupTask.draft_id == draft_id).first()
        if task:
            if task.status != "draft":
                return {"error": "invalid_state", "message": f"Cannot cancel: already {task.status}."}
            task.status = "cancelled"
            session.commit()
            return {"status": "cancelled", "message": "Action cancelled."}

        return {"error": "not_found", "message": "Draft not found."}
