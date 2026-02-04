"""
Prometheus Metrics Endpoint for Agentic RAG Backend.

PURPOSE:
    Expose /metrics endpoint for Alloy (Prometheus scraper) to collect metrics.

DATA FLOW:
    observability/metrics.py         This file                    Observability Stack
    ────────────────────────         ─────────                    ───────────────────
    Define & record metrics ──────►  /metrics endpoint ──────────► Alloy ──► Mimir ──► Grafana

AFTER THIS FILE:
    Register this router in fastapi_app.py:
        from src.presentation.api.metrics import router as metrics_router
        app.include_router(metrics_router)

    Test with: curl http://localhost:5001/metrics
"""

from fastapi import APIRouter, Response
from src.observability.metrics import get_metrics_content


router = APIRouter(prefix="/metrics", tags=["observability"])


@router.get("")
async def metrics():
    """
    Prometheus metrics endpoint.

    Alloy scrapes this endpoint every 15 seconds to collect metrics.
    Returns metrics in Prometheus text format.
    """
    content, content_type = get_metrics_content()
    return Response(content=content, media_type=content_type)
