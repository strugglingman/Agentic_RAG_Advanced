"""
Batch API Test Runner - Test backend chat API like frontend does.

This script:
1. Generates a valid JWT token for authentication
2. Sends queries to the backend /chat/agent endpoint (same as frontend)
3. Parses streaming response including __CONTEXT__ data
4. Analyzes results: answer quality, sources, decomposition usage
5. Exports results to JSON for analysis

Usage:
    python -m src.evaluation.batch_api_test --data eval_data/test.jsonl
    python -m src.evaluation.batch_api_test --query "Compare Delta and Starbucks revenue"
    python -m src.evaluation.batch_api_test --data eval_data/test.jsonl --limit 5
    python -m src.evaluation.batch_api_test --data eval_data/test.jsonl --reuse-conversation
    python -m src.evaluation.batch_api_test --data eval_data/test.jsonl --query-index 8

Environment:
    Requires backend to be running at BACKEND_URL (default: http://localhost:5001)
"""

import argparse
import json
import os
import sys
import time
import jwt
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional

import httpx

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Fix Windows encoding issues
from src.utils.encoding_fix import fix_windows_encoding
fix_windows_encoding()

from src.config.settings import Config


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class TestCase:
    """Single test case input."""

    question: str
    ground_truth: str = ""
    expected_sources: list[str] = field(default_factory=list)


@dataclass
class Context:
    """Retrieved context from backend."""

    chunk: str
    source: str
    page: int = 0
    chunk_id: str = ""
    file_id: str = ""
    sub_query: str = ""  # For decomposition
    hybrid: Optional[float] = None
    rerank: Optional[float] = None


@dataclass
class TestResult:
    """Result of a single test."""

    question: str
    ground_truth: str
    answer: str
    contexts: list[dict]
    conversation_id: str
    elapsed_time: float
    success: bool
    error: Optional[str] = None

    # Analysis fields
    num_contexts: int = 0
    sources_used: list[str] = field(default_factory=list)
    sub_queries_detected: list[str] = field(default_factory=list)
    has_decomposition: bool = False


# =============================================================================
# TOKEN GENERATION
# =============================================================================


def generate_test_token(
    email: str = "test@evaluation.com",
    dept_id: str = "EVAL|test|evaluation",
    expires_in: int = 3600,
) -> str:
    """
    Generate a valid JWT token for testing.

    Uses same secret/issuer/audience as backend expects.
    """
    now = datetime.utcnow()
    payload = {
        "email": email,
        "dept": dept_id,
        "iat": now,
        "exp": now + timedelta(seconds=expires_in),
        "iss": Config.SERVICE_AUTH_ISSUER,
        "aud": Config.SERVICE_AUTH_AUDIENCE,
    }

    token = jwt.encode(payload, Config.SERVICE_AUTH_SECRET, algorithm="HS256")
    return token


# =============================================================================
# API CLIENT
# =============================================================================


def parse_streaming_response(response_text: str) -> tuple[str, list[dict]]:
    """
    Parse streaming response from backend.

    Response format:
        <answer text>
        __CONTEXT__:<json array of contexts>

    Returns:
        Tuple of (answer, contexts)
    """
    answer = ""
    contexts = []

    if "__CONTEXT__:" in response_text:
        parts = response_text.split("__CONTEXT__:", 1)
        answer = parts[0].strip()
        try:
            context_json = parts[1].strip()
            contexts = json.loads(context_json)
        except json.JSONDecodeError as e:
            print(f"[WARN] Failed to parse contexts: {e}")
            contexts = []
    else:
        answer = response_text.strip()

    return answer, contexts


def send_query(
    client: httpx.Client,
    base_url: str,
    token: str,
    question: str,
    conversation_id: Optional[str] = None,
    timeout: int = 120,
) -> TestResult:
    """
    Send a query to the backend chat API.

    Mimics exactly what frontend does.
    """
    start_time = time.time()

    # Build request payload (same as frontend)
    payload = {
        "messages": [{"role": "user", "content": question}],
        "conversation_id": conversation_id,
        "filters": None,
        "attachments": None,
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    try:
        # Use streaming request
        with client.stream(
            "POST",
            f"{base_url}/chat/agent",
            json=payload,
            headers=headers,
            timeout=timeout,
        ) as response:
            response.raise_for_status()

            # Collect full response
            full_response = ""
            for chunk in response.iter_text():
                full_response += chunk

            # Get conversation ID from header
            conv_id = response.headers.get("X-Conversation-Id", "")

        elapsed = time.time() - start_time

        # Parse response
        answer, contexts = parse_streaming_response(full_response)

        # Analyze contexts
        sources = list(set(c.get("source", "") for c in contexts if c.get("source")))
        sub_queries = list(
            set(c.get("sub_query", "") for c in contexts if c.get("sub_query"))
        )
        has_decomposition = len(sub_queries) > 1

        return TestResult(
            question=question,
            ground_truth="",
            answer=answer,
            contexts=contexts,
            conversation_id=conv_id,
            elapsed_time=elapsed,
            success=True,
            num_contexts=len(contexts),
            sources_used=sources,
            sub_queries_detected=sub_queries,
            has_decomposition=has_decomposition,
        )

    except httpx.HTTPStatusError as e:
        elapsed = time.time() - start_time
        # Try to read error response body (may fail for streaming responses)
        try:
            error_body = e.response.text
        except Exception:
            error_body = "(response body not available)"

        # Build detailed error message
        error_details = [
            f"HTTP {e.response.status_code}: {error_body}",
            f"Request: POST {base_url}/chat/agent",
            f"Conversation ID: {conversation_id or 'None'}",
            f"Question length: {len(question)} chars",
        ]

        # Print detailed error to console for debugging
        print(f"\n  [ERROR DETAILS]")
        for detail in error_details:
            print(f"    {detail}")

        return TestResult(
            question=question,
            ground_truth="",
            answer="",
            contexts=[],
            conversation_id="",
            elapsed_time=elapsed,
            success=False,
            error=" | ".join(error_details),
        )
    except Exception as e:
        elapsed = time.time() - start_time
        # Get full traceback
        tb = traceback.format_exc()

        error_details = [
            f"Exception: {type(e).__name__}: {str(e)}",
            f"Conversation ID: {conversation_id or 'None'}",
            f"Question: {question[:100]}...",
        ]

        # Print detailed error to console
        print(f"\n  [ERROR DETAILS]")
        for detail in error_details:
            print(f"    {detail}")
        print(f"  [TRACEBACK]")
        print(f"    {tb}")

        return TestResult(
            question=question,
            ground_truth="",
            answer="",
            contexts=[],
            conversation_id="",
            elapsed_time=elapsed,
            success=False,
            error=f"{type(e).__name__}: {str(e)}",
        )


# =============================================================================
# TEST RUNNER
# =============================================================================


def load_test_cases(path: str) -> list[TestCase]:
    """Load test cases from JSONL file."""
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                cases.append(
                    TestCase(
                        question=data.get("question", ""),
                        ground_truth=data.get("ground_truth", ""),
                        expected_sources=data.get("expected_sources", []),
                    )
                )
    return cases


def run_batch_test(
    base_url: str,
    token: str,
    test_cases: list[TestCase],
    output_dir: str,
    verbose: bool = True,
    reuse_conversation: bool = False,
) -> list[TestResult]:
    """
    Run batch test against backend API.

    Args:
        reuse_conversation: If True, reuse same conversation_id for all queries
    """
    results = []
    conversation_id = None  # Track conversation across queries

    with httpx.Client() as client:
        for i, case in enumerate(test_cases, 1):
            if verbose:
                print(f"\n[{i}/{len(test_cases)}] Testing: {case.question[:60]}...")
                if reuse_conversation and conversation_id:
                    print(f"    → Using conversation: {conversation_id[:8]}...")

            result = send_query(
                client=client,
                base_url=base_url,
                token=token,
                question=case.question,
                conversation_id=conversation_id if reuse_conversation else None,
            )

            # Update conversation_id for next query if reusing
            if reuse_conversation and result.success and result.conversation_id:
                conversation_id = result.conversation_id

            # Add ground truth
            result.ground_truth = case.ground_truth

            if verbose:
                if result.success:
                    print(f"    ✓ Answer: {result.answer[:80]}...")
                    print(
                        f"    ✓ Contexts: {result.num_contexts}, Sources: {result.sources_used}"
                    )
                    print(f"    ✓ Time: {result.elapsed_time:.2f}s")
                    if result.has_decomposition:
                        print(f"    ✓ Decomposition: {result.sub_queries_detected}")
                    if reuse_conversation:
                        print(f"    ✓ Conversation ID: {result.conversation_id[:8]}...")
                else:
                    print(f"    ✗ Error: {result.error}")

            results.append(result)

    return results


def analyze_results(results: list[TestResult]) -> dict:
    """Analyze batch test results."""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    analysis = {
        "total_tests": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / len(results) if results else 0,
    }

    if successful:
        analysis["avg_time"] = sum(r.elapsed_time for r in successful) / len(successful)
        analysis["avg_contexts"] = sum(r.num_contexts for r in successful) / len(
            successful
        )
        analysis["decomposition_used"] = sum(
            1 for r in successful if r.has_decomposition
        )

        # Collect all unique sources
        all_sources = set()
        for r in successful:
            all_sources.update(r.sources_used)
        analysis["unique_sources"] = list(all_sources)

        # Count unique conversations
        unique_conversations = set(
            r.conversation_id for r in successful if r.conversation_id
        )
        analysis["unique_conversations"] = len(unique_conversations)

    return analysis


def export_results(
    results: list[TestResult],
    analysis: dict,
    output_path: str,
):
    """Export results to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "analysis": analysis,
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults exported to: {output_path}")


def print_summary(analysis: dict):
    """Print summary of test results."""
    print("\n" + "=" * 60)
    print("BATCH API TEST RESULTS")
    print("=" * 60)
    print(f"  Total tests: {analysis['total_tests']}")
    print(f"  Successful: {analysis['successful']}")
    print(f"  Failed: {analysis['failed']}")
    print(f"  Success rate: {analysis['success_rate']:.1%}")

    if analysis.get("avg_time"):
        print(f"  Avg response time: {analysis['avg_time']:.2f}s")
        print(f"  Avg contexts: {analysis['avg_contexts']:.1f}")
        print(f"  Decomposition used: {analysis['decomposition_used']} times")
        print(f"  Unique sources: {len(analysis.get('unique_sources', []))}")
        print(f"  Unique conversations: {analysis.get('unique_conversations', 0)}")

    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Batch test backend chat API like frontend"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:5001",
        help="Backend API base URL",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to test data JSONL file",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to test (alternative to --data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="src/evaluation/eval_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of tests (0 = all)",
    )
    parser.add_argument(
        "--query-index",
        type=int,
        default=0,
        help="Run only a specific query by index from JSONL (1-based index, requires --data)",
    )
    parser.add_argument(
        "--email",
        type=str,
        default="strugglingman@gmail.com",
        help="Test user email",
    )
    parser.add_argument(
        "--dept-id",
        type=str,
        default="MYHB|software|ml",
        help="Test department ID",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-test output",
    )
    parser.add_argument(
        "--reuse-conversation",
        action="store_true",
        help="Reuse same conversation for all queries (prevents creating many conversations)",
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.data and not args.query:
        print("Error: Must provide either --data or --query")
        sys.exit(1)

    if args.query_index > 0 and not args.data:
        print("Error: --query-index requires --data")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Generate token
    print("Generating authentication token...")
    token = generate_test_token(email=args.email, dept_id=args.dept_id)
    print(f"  Email: {args.email}")
    print(f"  Dept: {args.dept_id}")

    # Load test cases
    if args.query:
        test_cases = [TestCase(question=args.query)]
        print(f"\nSingle query mode: {args.query}")
    else:
        print(f"\nLoading test cases from: {args.data}")
        test_cases = load_test_cases(args.data)
        print(f"Loaded {len(test_cases)} test cases")

    # Select specific query if requested
    if args.query_index > 0:
        if args.query_index > len(test_cases):
            print(f"Error: Query index {args.query_index} exceeds total queries ({len(test_cases)})")
            sys.exit(1)
        selected_case = test_cases[args.query_index - 1]  # Convert to 0-based index
        test_cases = [selected_case]
        print(f"Selected query #{args.query_index}: {selected_case.question[:60]}...")

    if args.limit > 0 and args.query_index == 0:  # Only apply limit if not using query-index
        test_cases = test_cases[: args.limit]
        print(f"Limited to {len(test_cases)} test cases")

    # Run tests
    print(f"\nTesting against: {args.url}")
    if args.reuse_conversation:
        print("Mode: Reusing same conversation for all queries")
    print("-" * 60)

    results = run_batch_test(
        base_url=args.url,
        token=token,
        test_cases=test_cases,
        output_dir=args.output,
        verbose=not args.quiet,
        reuse_conversation=args.reuse_conversation,
    )

    # Analyze
    analysis = analyze_results(results)

    # Export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output, f"api_test_{timestamp}.json")
    export_results(results, analysis, output_path)

    # Print summary
    print_summary(analysis)

    # Print detailed results for single query
    if args.query and results and results[0].success:
        r = results[0]
        print("\n" + "=" * 60)
        print("DETAILED RESULT")
        print("=" * 60)
        print(f"\nQuestion: {r.question}")
        print(f"\nAnswer:\n{r.answer}")
        print(f"\nContexts ({r.num_contexts}):")
        for i, ctx in enumerate(r.contexts, 1):
            source = ctx.get("source", "unknown")
            page = ctx.get("page", 0)
            sub_q = ctx.get("sub_query", "")
            chunk = ctx.get("chunk", "")[:100]
            print(
                f"  [{i}] {source}"
                + (f" p.{page}" if page else "")
                + (f" (sub: {sub_q})" if sub_q else "")
            )
            print(f"      {chunk}...")
        print("=" * 60)


if __name__ == "__main__":
    main()
