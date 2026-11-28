import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import time
from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.health import router as health_router
from app.api.ws_text import router as ws_text_router
from app.api.ws_voice import router as ws_voice_router
from app.configs import PROJECT_ROOT, Settings, get_settings
from app.pipeline.latency import LatencyTracker
from app.pipeline.stt.base import create_stt_engine
from app.pipeline.tts.base import create_tts_engine
from app.pipeline.vad_energy import EnergyVAD

logger = logging.getLogger(__name__)


def load_agent_backend(backend: str):
    if backend == "stub":
        try:
            from dev.stub_agent import close_session, stream_agent_turn
            return stream_agent_turn, close_session
        except ImportError as exc:
            raise RuntimeError(
                "Stub agent backend unavailable: missing 'dev.stub_agent'."
            ) from exc
    if backend == "graph":
        try:
            from app.agent.graph import close_session, stream_agent_turn
            return stream_agent_turn, close_session
        except ImportError as exc:
            raise RuntimeError(
                "Graph agent backend unavailable: missing 'app.agent.graph'."
            ) from exc
    raise ValueError(f"Unknown agent_backend: {backend}")


def create_app(
    *,
    settings: Settings | None = None,
    stt: Any = None,
    tts: Any = None,
    stream_turn: Any = None,
    close_session_fn: Any = None,
    vad: Any = None,
    latency: LatencyTracker | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg = settings or get_settings()
        cfg.validate_for_runtime()

        Path(cfg.audit_log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.latency_log_path).parent.mkdir(parents=True, exist_ok=True)

        if int(os.environ.get("WEB_CONCURRENCY", "1")) > 1:
            raise RuntimeError(
                "Run with a single worker: sessions and models live in-process."
            )

        owned_stt = None
        if stt is None:
            owned_stt = create_stt_engine(cfg)
            t0 = time.perf_counter()
            await owned_stt.warmup()
            logger.info("STT warmup completed in %.2fs", time.perf_counter() - t0)
            app.state.stt = owned_stt
        else:
            app.state.stt = stt

        owned_tts = None
        if tts is None:
            owned_tts = create_tts_engine(cfg)
            t0 = time.perf_counter()
            await owned_tts.warmup()
            logger.info("TTS warmup completed in %.2fs", time.perf_counter() - t0)
            app.state.tts = owned_tts
        else:
            app.state.tts = tts

        owned_vad = None
        if vad is None:
            if cfg.vad_provider == "energy":
                owned_vad = EnergyVAD(cfg.audio_sample_rate)
                logger.info("Using lightweight energy-based VAD (no GPU required).")
            else:
                from app.pipeline.vad import SileroVAD
                owned_vad = SileroVAD(cfg.audio_sample_rate)
                logger.info("Using Silero VAD (GPU/CPU with torch).")
            app.state.vad = owned_vad
        else:
            app.state.vad = vad

        if stream_turn is None or close_session_fn is None:
            loaded_stream, loaded_close = load_agent_backend(cfg.agent_backend)
            app.state.stream_turn = stream_turn or loaded_stream
            app.state.close_session_fn = close_session_fn or loaded_close
        else:
            app.state.stream_turn = stream_turn
            app.state.close_session_fn = close_session_fn

        app.state.settings = cfg
        app.state.latency = latency or LatencyTracker(cfg.latency_log_path)
        app.state.session_lock = asyncio.Lock()

        try:
            yield
        finally:
            if owned_stt is not None:
                try:
                    await owned_stt.close()
                except Exception as exc:
                    logger.warning("Error closing STT engine: %s", exc)
            if owned_tts is not None:
                try:
                    await owned_tts.close()
                except Exception as exc:
                    logger.warning("Error closing TTS engine: %s", exc)
            if owned_vad is not None:
                app.state.vad = None

    app = FastAPI(title="Egyptian Voice Agent", lifespan=lifespan)
    app.include_router(health_router)
    app.include_router(ws_voice_router)
    app.include_router(ws_text_router)

    frontend_dir = PROJECT_ROOT / "frontend"
    if frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

    return app


app = create_app()