"""
LangGraph node smoke test runner.

Goal:
- Trigger plan-node-selected routes at least once per target tool route:
  direct_answer, retrieve, web_search, download_file, create_documents,
  send_email, code_execution.
- Capture route evidence from backend logs via X-Correlation-ID.
- Export JSON + Markdown report.

Usage:
    python -m src.evaluation.langgraph_node_smoke --url http://127.0.0.1:5001 \
        --dept-id "EVAL|nodesmoke|20260325" --email "strugglingman@gmail.com"
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import jwt

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.settings import Config


@dataclass
class RouteCase:
    case_id: str
    target_route: str
    query: str
    needs_resume: bool = False


@dataclass
class CaseResult:
    case_id: str
    target_route: str
    query: str
    correlation_id: str
    status_code: int
    elapsed_s: float
    answer_preview: str
    contexts_count: int
    hitl: bool
    hitl_action: str
    resume_called: bool
    resume_status_code: int
    resume_elapsed_s: float
    route_plan_steps: list[str] = field(default_factory=list)
    route_matches: list[str] = field(default_factory=list)
    detected_nodes: list[str] = field(default_factory=list)
    route_hit: bool = False
    request_ok: bool = False
    error: str = ""
    log_lines: list[str] = field(default_factory=list)


DEFAULT_CASES: list[RouteCase] = [
    RouteCase(
        case_id="direct_answer",
        target_route="direct_answer",
        query="Explain photosynthesis in two short sentences.",
    ),
    RouteCase(
        case_id="retrieve",
        target_route="retrieve",
        query="From our internal uploaded document financial_doc_2.txt, summarize three key points.",
    ),
    RouteCase(
        case_id="web_search",
        target_route="web_search",
        query="What is the weather in Stockholm today? Use current web information.",
    ),
    RouteCase(
        case_id="download_file",
        target_route="download_file",
        query="Download this file and give me the file_id: https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
    ),
    RouteCase(
        case_id="create_documents",
        target_route="create_documents",
        query="Create a markdown document titled 'Node Smoke Report' with three bullet points: alpha, beta, gamma, and provide the download link.",
    ),
    RouteCase(
        case_id="send_email",
        target_route="send_email",
        query="Send an email to smoke-test@example.com with subject 'LangGraph Node Smoke' and body 'This is a smoke test message.'",
        needs_resume=True,
    ),
    RouteCase(
        case_id="code_execution",
        target_route="code_execution",
        query="Use python code execution to compute the mean, median, and standard deviation of [12, 15, 20, 22, 23].",
    ),
]


def generate_token(email: str, dept_id: str, expires_in: int = 3600) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "email": email,
        "dept": dept_id,
        "iat": now,
        "exp": now + timedelta(seconds=expires_in),
        "iss": Config.SERVICE_AUTH_ISSUER,
        "aud": Config.SERVICE_AUTH_AUDIENCE,
    }
    return jwt.encode(payload, Config.SERVICE_AUTH_SECRET, algorithm="HS256")


def parse_sse_stream(response_text: str) -> tuple[str, list[dict[str, Any]], Optional[dict[str, Any]]]:
    answer_parts: list[str] = []
    contexts: list[dict[str, Any]] = []
    hitl_data: Optional[dict[str, Any]] = None

    for event_str in response_text.split("\n\n"):
        if not event_str.strip():
            continue
        event_type = "message"
        data_lines: list[str] = []
        for line in event_str.split("\n"):
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                data_lines.append(line[6:])
        data = "\n".join(data_lines)
        if event_type == "text":
            answer_parts.append(data)
        elif event_type == "context":
            try:
                parsed = json.loads(data)
                if isinstance(parsed, list):
                    contexts = parsed
            except json.JSONDecodeError:
                pass
        elif event_type == "hitl":
            try:
                parsed_hitl = json.loads(data)
                if isinstance(parsed_hitl, dict):
                    hitl_data = parsed_hitl
            except json.JSONDecodeError:
                pass

    return "".join(answer_parts).strip(), contexts, hitl_data


def read_log_delta(log_path: Path, start_pos: int, correlation_id: str) -> list[str]:
    if not log_path.exists():
        return []
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(start_pos, 0)
        chunk = f.read()
    lines = [ln for ln in chunk.splitlines() if correlation_id in ln]
    return lines


def parse_route_evidence(log_lines: list[str]) -> tuple[list[str], list[str], list[str]]:
    plan_steps: list[str] = []
    route_matches: list[str] = []
    detected_nodes: set[str] = set()

    route_match_pattern = re.compile(r"Matched '([^']+)' -> ([a-z_]+)")

    for ln in log_lines:
        low = ln.lower()

        if "[plan] structured plan created:" in low:
            raw = ln.split("Structured plan created:", 1)[-1].strip()
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, list):
                    plan_steps.extend([str(x) for x in parsed])
            except Exception:
                plan_steps.append(raw)

        if "[route_after_planning]" in low and "matched" in low:
            m = route_match_pattern.search(ln)
            if m:
                planned_tool, target_node = m.group(1), m.group(2)
                route_matches.append(f"{planned_tool}->{target_node}")
                if planned_tool in {
                    "direct_answer",
                    "retrieve",
                    "web_search",
                    "download_file",
                    "create_documents",
                    "send_email",
                    "code_execution",
                }:
                    detected_nodes.add(planned_tool)

        if "[retrieve]" in low:
            detected_nodes.add("retrieve")
        if "[web_search]" in low or "tool_web_search" in low:
            detected_nodes.add("web_search")
        if "[download_file_node]" in low or "tool_download_file" in low:
            detected_nodes.add("download_file")
        if "[create_documents_node]" in low or "tool_create_documents" in low:
            detected_nodes.add("create_documents")
        if "[send_email_node]" in low or "tool_send_email" in low:
            detected_nodes.add("send_email")
        if "[code_execution_node]" in low or "[code_execution]" in low or "tool_code_execution" in low:
            detected_nodes.add("code_execution")
        if "generated answer from direct_answer directly" in low or "direct_answer" in low and "route_after_planning" in low:
            detected_nodes.add("direct_answer")

    return plan_steps, route_matches, sorted(detected_nodes)


def post_stream(
    client: httpx.Client,
    url: str,
    token: str,
    payload: dict[str, Any],
    correlation_id: str,
    timeout_s: float,
) -> tuple[int, dict[str, str], str, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Correlation-ID": correlation_id,
    }
    with client.stream(
        "POST",
        url,
        json=payload,
        headers=headers,
        timeout=timeout_s,
    ) as resp:
        status = resp.status_code
        headers_out = {k: v for k, v in resp.headers.items()}
        text_body = "".join(resp.iter_text())
        return status, headers_out, text_body, resp.reason_phrase


def run_case(
    client: httpx.Client,
    base_url: str,
    token: str,
    case: RouteCase,
    log_path: Path,
    timeout_s: float,
) -> CaseResult:
    correlation_id = f"node-smoke-{case.case_id}-{uuid.uuid4().hex[:8]}"
    log_start = log_path.stat().st_size if log_path.exists() else 0

    start = time.time()
    status_code = 0
    answer_preview = ""
    contexts_count = 0
    hitl = False
    hitl_action = ""
    resume_called = False
    resume_status_code = 0
    resume_elapsed_s = 0.0
    error = ""
    combined_log_lines: list[str] = []

    try:
        chat_payload = {
            "messages": [{"role": "user", "content": case.query}],
            "conversation_id": None,
            "filters": None,
            "attachments": None,
        }
        status_code, headers_out, body, _ = post_stream(
            client=client,
            url=f"{base_url}/chat/agent",
            token=token,
            payload=chat_payload,
            correlation_id=correlation_id,
            timeout_s=timeout_s,
        )

        answer, contexts, hitl_data = parse_sse_stream(body)
        answer_preview = answer[:220]
        contexts_count = len(contexts)
        hitl = hitl_data is not None
        hitl_action = (hitl_data or {}).get("action", "")
        conv_id = headers_out.get("x-conversation-id", "")

        # Resume send_email HITL if requested and available
        if case.needs_resume and hitl_data and hitl_data.get("thread_id"):
            resume_called = True
            resume_corr = f"{correlation_id}-resume"
            resume_log_start = log_path.stat().st_size if log_path.exists() else 0
            resume_start = time.time()
            resume_payload = {
                "thread_id": hitl_data["thread_id"],
                "confirmed": True,
                "conversation_id": conv_id or None,
            }
            resume_status_code, _, resume_body, _ = post_stream(
                client=client,
                url=f"{base_url}/chat/resume",
                token=token,
                payload=resume_payload,
                correlation_id=resume_corr,
                timeout_s=timeout_s,
            )
            resume_elapsed_s = time.time() - resume_start
            resume_answer, _, _ = parse_sse_stream(resume_body)
            if resume_answer:
                answer_preview = (answer_preview + " | RESUME: " + resume_answer[:140])[:220]
            time.sleep(0.8)
            combined_log_lines.extend(read_log_delta(log_path, resume_log_start, resume_corr))

        time.sleep(0.8)
        combined_log_lines.extend(read_log_delta(log_path, log_start, correlation_id))

    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        time.sleep(0.8)
        combined_log_lines.extend(read_log_delta(log_path, log_start, correlation_id))

    elapsed_s = time.time() - start
    plan_steps, route_matches, detected_nodes = parse_route_evidence(combined_log_lines)

    route_hit = case.target_route in detected_nodes
    request_ok = status_code == 200 and not error
    if case.needs_resume and resume_called:
        request_ok = request_ok and resume_status_code == 200

    return CaseResult(
        case_id=case.case_id,
        target_route=case.target_route,
        query=case.query,
        correlation_id=correlation_id,
        status_code=status_code,
        elapsed_s=round(elapsed_s, 2),
        answer_preview=answer_preview,
        contexts_count=contexts_count,
        hitl=hitl,
        hitl_action=hitl_action,
        resume_called=resume_called,
        resume_status_code=resume_status_code,
        resume_elapsed_s=round(resume_elapsed_s, 2),
        route_plan_steps=plan_steps,
        route_matches=route_matches,
        detected_nodes=detected_nodes,
        route_hit=route_hit,
        request_ok=request_ok,
        error=error,
        log_lines=combined_log_lines[-120:],
    )


def write_markdown_report(
    results: list[CaseResult],
    output_md: Path,
    base_url: str,
    dept_id: str,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    lines: list[str] = [
        "# LangGraph Node Smoke Report",
        "",
        f"- Timestamp (UTC): `{timestamp}`",
        f"- Backend URL: `{base_url}`",
        f"- Dept ID: `{dept_id}`",
        "",
        "## Summary",
        "",
        "| Case | Target Route | Route Hit | Request OK | Status | HITL | Resume | Elapsed(s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for r in results:
        lines.append(
            f"| {r.case_id} | {r.target_route} | {'Y' if r.route_hit else 'N'} | "
            f"{'Y' if r.request_ok else 'N'} | {r.status_code} | "
            f"{'Y' if r.hitl else 'N'} | {'Y' if r.resume_called else 'N'} | {r.elapsed_s} |"
        )

    lines.extend(
        [
            "",
            "## Detailed Results",
            "",
        ]
    )

    for r in results:
        lines.extend(
            [
                f"### {r.case_id}",
                "",
                f"- Target route: `{r.target_route}`",
                f"- Query: `{r.query}`",
                f"- Correlation ID: `{r.correlation_id}`",
                f"- HTTP status: `{r.status_code}`",
                f"- Route hit: `{r.route_hit}`",
                f"- Request OK: `{r.request_ok}`",
                f"- Plan steps: `{r.route_plan_steps}`",
                f"- Route matches: `{r.route_matches}`",
                f"- Detected nodes: `{r.detected_nodes}`",
                f"- HITL: `{r.hitl}` action=`{r.hitl_action}`",
                f"- Resume: called=`{r.resume_called}` status=`{r.resume_status_code}` elapsed=`{r.resume_elapsed_s}`",
                f"- Context count: `{r.contexts_count}`",
                f"- Answer preview: `{r.answer_preview}`",
            ]
        )
        if r.error:
            lines.append(f"- Error: `{r.error}`")
        lines.append("")

    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="LangGraph node smoke coverage runner")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:5001")
    parser.add_argument("--email", type=str, default="strugglingman@gmail.com")
    parser.add_argument("--dept-id", type=str, default="EVAL|nodesmoke|20260325")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-request timeout seconds")
    parser.add_argument("--log-path", type=str, default="logs/app.log")
    parser.add_argument("--output", type=str, default="src/evaluation/eval_results")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path)

    token = generate_token(args.email, args.dept_id)
    print(f"[INFO] Running LangGraph node smoke tests against: {args.url}")
    print(f"[INFO] Dept: {args.dept_id}, Email: {args.email}")
    print(f"[INFO] Log path: {log_path}")

    results: list[CaseResult] = []
    with httpx.Client() as client:
        for case in DEFAULT_CASES:
            print(f"\n[CASE] {case.case_id} -> target={case.target_route}")
            result = run_case(
                client=client,
                base_url=args.url,
                token=token,
                case=case,
                log_path=log_path,
                timeout_s=args.timeout,
            )
            results.append(result)
            print(
                f"  route_hit={result.route_hit}, request_ok={result.request_ok}, "
                f"status={result.status_code}, elapsed={result.elapsed_s}s, "
                f"nodes={result.detected_nodes}"
            )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = output_dir / f"langgraph_node_smoke_{ts}.json"
    output_md = output_dir / f"langgraph_node_smoke_{ts}.md"

    output_json.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "backend_url": args.url,
                "dept_id": args.dept_id,
                "results": [asdict(r) for r in results],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_markdown_report(results, output_md, args.url, args.dept_id)

    covered = sorted({n for r in results for n in r.detected_nodes})
    expected = sorted({c.target_route for c in DEFAULT_CASES})
    missing = [x for x in expected if x not in covered]
    print("\n" + "=" * 72)
    print("LANGGRAPH NODE SMOKE SUMMARY")
    print("=" * 72)
    print(f"Expected routes: {expected}")
    print(f"Detected routes: {covered}")
    print(f"Missing routes:  {missing if missing else 'None'}")
    print(f"JSON report: {output_json}")
    print(f"MD report:   {output_md}")
    print("=" * 72)

    # Return non-zero if any target route not hit
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
