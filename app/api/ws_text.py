import contextlib
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.contracts import AgentRequest, SessionContext

router = APIRouter()


@router.websocket("/ws/text")
async def ws_text(websocket: WebSocket) -> None:
    await websocket.accept()
    state = websocket.app.state
    seen_sessions: set[str] = set()

    try:
        while True:
            data = await websocket.receive_json()
            if not isinstance(data, dict):
                continue
            text = str(data.get("text", ""))
            session_id = str(data.get("session_id") or uuid4().hex)
            seen_sessions.add(session_id)
            email = data.get("email")
            ctx = SessionContext(user_email=email) if isinstance(email, str) and email else None

            req = AgentRequest(session_id=session_id, user_text=text, context=ctx)
            async for chunk in state.stream_turn(req):
                await websocket.send_json(chunk.model_dump(exclude_none=True))
    except WebSocketDisconnect:
        pass
    finally:
        for sid in seen_sessions:
            with contextlib.suppress(Exception):
                await state.close_session_fn(sid)
