"""
Batch Slack Test Runner - Test Slack adapter with simulated events.

This script:
1. Loads test cases from JSONL files (same format as batch_api_test.py)
2. Creates mock Slack events
3. Calls SlackBotAdapter.handle_message() directly
4. Captures responses via mocked Slack API
5. Analyzes results and exports to JSON

Usage:
    python -m src.evaluation.batch_slack_test --data eval_data/financial_test.jsonl
    python -m src.evaluation.batch_slack_test --query "What is NVIDIA's revenue?"
    python -m src.evaluation.batch_slack_test --data eval_data/financial_test.jsonl --limit 5

Environment:
    Requires backend to be running at BOT_BACKEND_URL (default: http://localhost:5001)
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

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
class SlackTestResult:
    """Result of a single Slack test."""
    question: str
    ground_truth: str
    answer: str
    contexts: list[dict]
    elapsed_time: float
    success: bool
    error: Optional[str] = None

    # Slack-specific fields
    channel_id: str = ""
    message_ts: str = ""
    history_messages_used: int = 0

    # Analysis fields
    num_contexts: int = 0
    sources_used: list[str] = field(default_factory=list)


# =============================================================================
# MOCK SLACK CLIENT
# =============================================================================


class MockSlackClient:
    """Mock Slack client that captures responses."""

    def __init__(self):
        self.messages_sent: list[dict] = []
        self.reactions_added: list[dict] = []
        self.reactions_removed: list[dict] = []
        self.files_uploaded: list[dict] = []
        self.last_response: Optional[dict] = None

    async def chat_postMessage(self, **kwargs):
        """Capture posted messages."""
        self.messages_sent.append(kwargs)
        self.last_response = kwargs
        return {"ok": True, "ts": f"mock_{time.time()}"}

    async def chat_update(self, **kwargs):
        """Capture message updates."""
        self.messages_sent.append({"_update": True, **kwargs})
        return {"ok": True}

    async def reactions_add(self, **kwargs):
        """Capture reaction adds."""
        self.reactions_added.append(kwargs)
        return {"ok": True}

    async def reactions_remove(self, **kwargs):
        """Capture reaction removes."""
        self.reactions_removed.append(kwargs)
        return {"ok": True}

    async def files_upload_v2(self, **kwargs):
        """Capture file uploads."""
        self.files_uploaded.append(kwargs)
        return {"ok": True}

    async def conversations_history(self, **kwargs):
        """Return empty history (can be customized)."""
        return {"messages": [], "ok": True}

    async def users_info(self, user: str):
        """Return mock user info."""
        return {
            "ok": True,
            "user": {
                "id": user,
                "profile": {
                    "email": "test@example.com",
                    "display_name": "Test User",
                    "real_name": "Test User",
                }
            }
        }

    def get_answer_text(self) -> str:
        """Extract answer text from last response."""
        if not self.last_response:
            return ""

        blocks = self.last_response.get("blocks", [])
        texts = []
        for block in blocks:
            if block.get("type") == "section":
                text_obj = block.get("text", {})
                if text_obj.get("type") == "mrkdwn":
                    texts.append(text_obj.get("text", ""))

        return "\n".join(texts)

    def reset(self):
        """Reset captured data."""
        self.messages_sent = []
        self.reactions_added = []
        self.reactions_removed = []
        self.files_uploaded = []
        self.last_response = None


# =============================================================================
# TEST RUNNER
# =============================================================================


def create_mock_slack_event(
    text: str,
    user_id: str = "U_TEST_USER",
    channel_id: str = "C_TEST_CHANNEL",
    ts: Optional[str] = None,
) -> dict:
    """Create a mock Slack message event."""
    return {
        "type": "message",
        "text": text,
        "user": user_id,
        "channel": channel_id,
        "ts": ts or str(time.time()),
        "team": "T_TEST_TEAM",
        "files": [],
    }


async def run_single_test(
    adapter,
    mock_client: MockSlackClient,
    test_case: TestCase,
    channel_id: str = "C_TEST_CHANNEL",
) -> SlackTestResult:
    """Run a single test case through the Slack adapter."""

    start_time = time.time()
    mock_client.reset()

    # Create mock event
    event = create_mock_slack_event(
        text=test_case.question,
        channel_id=channel_id,
    )

    try:
        # Run through adapter
        await adapter.handle_message(event)

        elapsed = time.time() - start_time

        # Extract answer from mock client
        answer = mock_client.get_answer_text()

        # Try to extract contexts from the response
        contexts = []
        # Note: contexts are typically in the BotResponse, but we don't have direct access
        # We could parse from the formatted message if needed

        # Count sources from message (if citations are included)
        sources = []
        for msg in mock_client.messages_sent:
            blocks = msg.get("blocks", [])
            for block in blocks:
                if block.get("type") == "context":
                    elements = block.get("elements", [])
                    for elem in elements:
                        text = elem.get("text", "")
                        if "Sources:" in text:
                            # Extract source filenames
                            source_text = text.replace(":page_facing_up:", "").replace("*Sources:*", "").strip()
                            sources = [s.strip() for s in source_text.split(",")]

        return SlackTestResult(
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            answer=answer,
            contexts=contexts,
            elapsed_time=elapsed,
            success=True,
            channel_id=channel_id,
            message_ts=event["ts"],
            num_contexts=len(contexts),
            sources_used=sources,
        )

    except Exception as e:
        elapsed = time.time() - start_time
        import traceback
        tb = traceback.format_exc()
        print(f"  [ERROR] {type(e).__name__}: {e}")
        print(f"  [TRACEBACK]\n{tb}")

        return SlackTestResult(
            question=test_case.question,
            ground_truth=test_case.ground_truth,
            answer="",
            contexts=[],
            elapsed_time=elapsed,
            success=False,
            error=f"{type(e).__name__}: {str(e)}",
            channel_id=channel_id,
            message_ts=event.get("ts", ""),
        )


async def run_batch_test(
    test_cases: list[TestCase],
    verbose: bool = True,
) -> list[SlackTestResult]:
    """Run batch test through Slack adapter."""

    # Import adapter here to avoid circular imports
    from src.adapters.slack.slack_adapter import SlackBotAdapter

    # Create mock client
    mock_client = MockSlackClient()

    # Create adapter with mocked client
    adapter = SlackBotAdapter()
    adapter._client = mock_client

    # Mock identity resolver to return test user
    async def mock_resolve_identity(event):
        from src.adapters.base_bot_adapter import BotUser
        return BotUser(
            user_email="test@evaluation.com",
            dept_id="EVAL|test|evaluation",
            platform_user_id=event.get("user", "U_TEST"),
            display_name="Test User",
        )

    adapter._identity_resolver.resolve_identity = mock_resolve_identity
    adapter._identity_resolver.get_display_name = AsyncMock(return_value="Test User")

    results = []

    for i, case in enumerate(test_cases, 1):
        if verbose:
            print(f"\n[{i}/{len(test_cases)}] Testing: {case.question[:60]}...")

        result = await run_single_test(
            adapter=adapter,
            mock_client=mock_client,
            test_case=case,
        )

        if verbose:
            if result.success:
                print(f"    ✓ Answer: {result.answer[:80]}...")
                print(f"    ✓ Time: {result.elapsed_time:.2f}s")
                if result.sources_used:
                    print(f"    ✓ Sources: {result.sources_used}")
            else:
                print(f"    ✗ Error: {result.error}")

        results.append(result)

    return results


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


def analyze_results(results: list[SlackTestResult]) -> dict:
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
        analysis["avg_contexts"] = sum(r.num_contexts for r in successful) / len(successful)

        # Collect all unique sources
        all_sources = set()
        for r in successful:
            all_sources.update(r.sources_used)
        analysis["unique_sources"] = list(all_sources)

    return analysis


def export_results(
    results: list[SlackTestResult],
    analysis: dict,
    output_path: str,
):
    """Export results to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "slack_batch_test",
        "analysis": analysis,
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults exported to: {output_path}")


def print_summary(analysis: dict):
    """Print summary of test results."""
    print("\n" + "=" * 60)
    print("BATCH SLACK TEST RESULTS")
    print("=" * 60)
    print(f"  Total tests: {analysis['total_tests']}")
    print(f"  Successful: {analysis['successful']}")
    print(f"  Failed: {analysis['failed']}")
    print(f"  Success rate: {analysis['success_rate']:.1%}")

    if analysis.get("avg_time"):
        print(f"  Avg response time: {analysis['avg_time']:.2f}s")
        print(f"  Avg contexts: {analysis['avg_contexts']:.1f}")
        print(f"  Unique sources: {len(analysis.get('unique_sources', []))}")

    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Batch test Slack adapter with simulated events"
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
        help="Run only a specific query by index (1-based)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-test output",
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.data and not args.query:
        print("Error: Must provide either --data or --query")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

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
            print(f"Error: Query index {args.query_index} exceeds total ({len(test_cases)})")
            sys.exit(1)
        test_cases = [test_cases[args.query_index - 1]]
        print(f"Selected query #{args.query_index}")

    if args.limit > 0 and args.query_index == 0:
        test_cases = test_cases[:args.limit]
        print(f"Limited to {len(test_cases)} test cases")

    # Run tests
    print(f"\nTesting Slack adapter against backend: {Config.BOT_BACKEND_URL}")
    print("-" * 60)

    results = asyncio.run(run_batch_test(
        test_cases=test_cases,
        verbose=not args.quiet,
    ))

    # Analyze
    analysis = analyze_results(results)

    # Export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output, f"slack_test_{timestamp}.json")
    export_results(results, analysis, output_path)

    # Print summary
    print_summary(analysis)

    # Print detailed result for single query
    if args.query and results and results[0].success:
        r = results[0]
        print("\n" + "=" * 60)
        print("DETAILED RESULT")
        print("=" * 60)
        print(f"\nQuestion: {r.question}")
        print(f"\nAnswer:\n{r.answer}")
        print("=" * 60)


if __name__ == "__main__":
    main()