"""Factory aplikasi FastAPI untuk prototype demo lokal attire-compliance."""

from __future__ import annotations

from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from attire_compliance.app.routes import router
from attire_compliance.pipeline.pipeline import AttireCompliancePipeline

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
TEMPLATE_PATH = APP_DIR / "templates" / "index.html"


def create_app(*, pipeline_override: Any | None = None, warmup: bool = True) -> FastAPI:
    """Bangun instance FastAPI dengan single-worker GPU executor dan asset 100% lokal."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="attire-gpu")
        app.state.executor = executor
        app.state.template_path = TEMPLATE_PATH

        if pipeline_override is not None:
            app.state.pipeline = pipeline_override
        else:
            pipeline = AttireCompliancePipeline()
            if warmup:
                pipeline.warmup()
            app.state.pipeline = pipeline

        try:
            yield
        finally:
            executor.shutdown(wait=False)

    app = FastAPI(
        title="Sistem Deteksi Etika Berpakaian Mahasiswa (YOLOv12 + EfficientNetV2)",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(router)
    return app


app = create_app()
