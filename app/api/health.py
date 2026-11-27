from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    state = request.app.state
    settings = state.settings
    return {
        "status": "ok",
        "stt": {
            "provider": settings.stt_provider,
            "device": getattr(state.stt, "device", None),
        },
        "tts": {
            "provider": settings.tts_provider,
            "device": getattr(state.tts, "device", None),
            "sample_rate": getattr(state.tts, "sample_rate", None),
        },
        "agent_backend": settings.agent_backend,
        "dry_run": settings.dry_run,
        "active_session": state.session_lock.locked(),
    }


@router.get("/metrics")
async def metrics(request: Request) -> dict:
    return request.app.state.latency.summary()
