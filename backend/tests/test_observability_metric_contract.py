import json
import os
import re
from pathlib import Path

import pytest

from src.observability import metrics as backend_metrics


def _resolve_observability_root() -> Path | None:
    candidates = [
        Path(os.getenv("OBSERVABILITY_REPO_PATH", "")),
        Path("d:/agentic_rag_observability"),
        Path(__file__).resolve().parents[2] / "agentic_rag_observability",
    ]
    for candidate in candidates:
        if candidate and str(candidate) and (candidate / "grafana" / "dashboards").exists():
            return candidate
    return None


def _normalize_metric_name(name: str) -> str:
    for suffix in ("_bucket", "_sum", "_count", "_created"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _extract_dashboard_metrics(expr: str) -> set[str]:
    metrics = set()
    for match in re.findall(r"(?<![A-Za-z0-9_])([a-zA-Z_:][a-zA-Z0-9_:]*)(?=(\{|\[))", expr):
        metrics.add(match[0])

    expr_trim = expr.strip()
    if re.fullmatch(r"[a-zA-Z_:][a-zA-Z0-9_:]*", expr_trim):
        metrics.add(expr_trim)
    return metrics


def _backend_metric_names() -> set[str]:
    collected = set()
    for attr in (
        "ACTIVE_QUERIES",
        "REQUEST_LATENCY",
        "RETRIEVAL_LATENCY",
        "LLM_TOKENS_TOTAL",
        "QUERY_ROUTING_TOTAL",
        "SELF_REFLECTION_TOTAL",
        "CHUNK_RELEVANCE_SCORE",
        "ERRORS_TOTAL",
    ):
        metric_obj = getattr(backend_metrics, attr)
        collected.add(metric_obj._name)  # prometheus_client metric name
    return collected


def test_dashboard_metrics_exist_in_backend_exporters():
    obs_root = _resolve_observability_root()
    if obs_root is None:
        pytest.skip("Observability repo not found in expected locations")

    dashboard_path = obs_root / "grafana" / "dashboards" / "rag-overview.json"
    data = json.loads(dashboard_path.read_text(encoding="utf-8"))

    referenced = set()
    for panel in data.get("panels", []):
        for target in panel.get("targets", []):
            expr = target.get("expr")
            if not expr:
                continue
            referenced.update(_extract_dashboard_metrics(expr))

    normalized_referenced = {_normalize_metric_name(name) for name in referenced}
    backend_names = _backend_metric_names()

    missing = sorted(normalized_referenced - backend_names)
    assert missing == [], f"Dashboard references undefined backend metrics: {missing}"
