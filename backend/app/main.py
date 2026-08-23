import json
import uuid
import asyncio
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from app.auth import get_user_by_token, UserContext
from app.agent.graph import graph
from langgraph.types import Command
from app.tools.actions import execute_action, cancel_action
from app.ops.detectors import run_detectors

app = FastAPI(title="ParcelPilot Copilot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


def get_current_user(token: str) -> UserContext:
    user = get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user


class LoginRequest(BaseModel):
    user_key: str


@app.post("/auth/login")
async def login(req: LoginRequest):
    user = get_user_by_token(req.user_key)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return {"token": user.user_key, "user": user.model_dump()}


class ChatRequest(BaseModel):
    message: str
    token: str
    thread_id: str | None = None  # client can pass to continue a conversation


# Track active thread_ids for confirm flow
_active_threads: dict[str, str] = {}  # user_key -> last thread_id


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    user = get_current_user(req.token)

    # Each message gets a fresh thread_id to avoid state bleed
    # Unless client explicitly passes one (for confirm flow)
    thread_id = req.thread_id or str(uuid.uuid4())
    _active_threads[user.user_key] = thread_id

    async def event_generator():
        config = {
            "configurable": {
                "user_context": user,
                "thread_id": thread_id,
            }
        }
        input_state = {
            "messages": [("user", req.message)],
            "original_query": req.message,
            "intent": None,
            "entities": None,
            "records": None,
            "capabilities": None,
            "computed": None,
            "plan_trace": [],
            "retrieval_attempt": 0,
            "iterations": 0,
            "citations": [],
            "pending_draft": None,
            "resolution": None,
            "execute_result": None,
            "final_answer": None,
            "user_confirmation": None,
        }

        try:
            final_state = {}
            announced_intent = False

            async for event in graph.astream_events(input_state, config, version="v2"):
                kind = event["event"]
                node_name = event.get("metadata", {}).get("langgraph_node")

                if kind == "on_chat_model_stream":
                    # Only the ReAct fallback path streams prose to the bubble.
                    if node_name == "agent":
                        chunk = event["data"]["chunk"]
                        if chunk.content:
                            yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"
                elif kind == "on_tool_start":
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': event['name'], 'input': event['data'].get('input')})}\n\n"
                elif kind == "on_tool_end":
                    yield f"data: {json.dumps({'type': 'tool_end', 'tool': event['name']})}\n\n"
                elif kind == "on_chain_end":
                    output = event["data"].get("output")
                    if not (output and isinstance(output, dict)):
                        continue

                    for key in ("resolution", "pending_draft", "final_answer",
                                "intent", "computed", "plan_trace", "capabilities"):
                        if key in output:
                            final_state[key] = output[key]

                    if output.get("intent") and not announced_intent:
                        announced_intent = True
                        yield f"data: {json.dumps({'type': 'intent', 'intent': output['intent'], 'entities': output.get('entities')})}\n\n"

                    if node_name == "compute" and output.get("computed"):
                        yield f"data: {json.dumps({'type': 'computation', 'computed': output['computed']})}\n\n"

            if final_state.get("plan_trace"):
                yield f"data: {json.dumps({'type': 'plan_trace', 'steps': final_state['plan_trace']})}\n\n"
            yield f"data: {json.dumps({'type': 'resolution', 'resolution': final_state.get('resolution')})}\n\n"
            if final_state.get("pending_draft"):
                yield f"data: {json.dumps({'type': 'pending_action', 'draft': final_state.get('pending_draft'), 'thread_id': thread_id})}\n\n"
            if final_state.get("final_answer"):
                # final_answer is already valid JSON from Pydantic model_dump_json()
                # Send it as a parsed object so the frontend doesn't have to parse JSON from a string
                try:
                    parsed = json.loads(final_state["final_answer"])
                    yield f"data: {json.dumps({'type': 'final_answer', 'parsed': parsed})}\n\n"
                except (json.JSONDecodeError, TypeError):
                    yield f"data: {json.dumps({'type': 'final_answer', 'content': final_state['final_answer']})}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


class ConfirmRequest(BaseModel):
    draft_id: str
    confirm: bool
    token: str
    thread_id: str | None = None


@app.post("/chat/confirm")
async def confirm_action(req: ConfirmRequest):
    user = get_current_user(req.token)
    thread_id = req.thread_id or _active_threads.get(user.user_key)
    if not thread_id:
        raise HTTPException(status_code=400, detail="No active thread found for confirmation.")

    config = {"configurable": {"user_context": user, "thread_id": thread_id}}
    try:
        # Resume the graph with the user's confirmation decision
        state = graph.invoke(Command(resume={"confirm": req.confirm}), config)
        final_answer = state.get("final_answer", "")
        if not final_answer:
            if req.confirm:
                final_answer = "Action executed successfully."
            else:
                final_answer = "Action cancelled."
        return {"status": "success", "message": final_answer}
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Fallback: execute/cancel directly
        if req.confirm:
            res = execute_action(req.draft_id, user)
        else:
            res = cancel_action(req.draft_id, user)
        return {"status": res.get("status", "error"), "message": res.get("message", str(e))}


@app.get("/ops/issues")
async def ops_issues(token: str):
    user = get_current_user(token)
    if user.role == "customer":
        raise HTTPException(status_code=403, detail="Access denied. Internal users only.")
    return run_detectors(user)


@app.post("/ops/scan")
async def ops_scan(token: str):
    user = get_current_user(token)
    if user.role == "customer":
        raise HTTPException(status_code=403, detail="Access denied. Internal users only.")
    return run_detectors(user)


@app.get("/health")
async def health():
    return {"status": "ok", "snapshot": "2026-08-16T11:00:00+05:30"}


@app.get("/")
async def index():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return {"message": "ParcelPilot Copilot API. Frontend not found at expected path."}
