"""
Test script for the Agent service (ReAct pattern).

This script tests:
1. Calculator tool (simple tool call)
2. Document search tool (integration with RAG)
3. Multi-step reasoning (chained tool calls)
4. Error handling

HOW TO RUN:
-----------
From the backend directory:
    python tests/test_agent_service.py

Or with pytest:
    pytest tests/test_agent_service.py -v

IMPORTANT - Testing Document Search:
------------------------------------
Tests 3 & 4 (document_search and multi_step) require documents in ChromaDB.

If you get "No documents in collection" warnings:

Option 1 - Use the Web UI (Recommended):
    1. Start the full app: python run.py
    2. Go to http://localhost:5001
    3. Login and upload some PDF/TXT documents
    4. Click "Ingest" to add them to the vector DB
    5. Stop the app and run this test script

Option 2 - Add test documents manually:
    1. Place some .txt or .pdf files in backend/test_data/
    2. Use the ingest script to add them to ChromaDB
    3. Run this test script

Tests 1, 2, 5, 6 don't need documents and will work immediately.
"""

import os
import sys

# Add backend directory to Python path so we can import from src
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import chromadb
from chromadb.utils.embedding_functions import sentence_transformer_embedding_function
from openai import OpenAI
from src.services.agent_service import AgentService
from src.config.settings import Config
from flask import request


# Color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_test(test_name):
    """Print test name"""
    print(f"{Colors.OKCYAN}{Colors.BOLD}Test: {test_name}{Colors.ENDC}")


def print_success(message):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def print_error(message):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def print_result(label, value):
    """Print a result"""
    print(f"{Colors.OKBLUE}{label}:{Colors.ENDC} {value}")


class MockRequest:
    def __init__(self):
        self.json = {}

    def get_json(self, force=True):
        return self.json


def get_collection():
    """
    Get or create ChromaDB collection.

    This initializes the collection the same way as app.py does.
    Returns the collection object that can be passed to the agent context.
    """
    embed_model_name = Config.EMBED_MODEL_NAME
    chroma_path = "../chroma_db"
    embedding_fun = (
        sentence_transformer_embedding_function.SentenceTransformerEmbeddingFunction(
            model_name=embed_model_name
        )
    )
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    collection = chroma_client.get_or_create_collection(
        name="docs", metadata={"hnsw:space": "cosine"}, embedding_function=embedding_fun
    )
    return collection


def test_calculator():
    """Test 1: Simple calculator tool"""
    print_test("Calculator Tool - Basic Math")

    # Initialize OpenAI client
    client = OpenAI(api_key=Config.OPENAI_KEY)

    # Initialize agent
    agent = AgentService(openai_client=client, max_iterations=5)

    try:
        # Test simple calculation
        answer = agent.run(query="What is 15% of 200?", context={})

        print_result("Query", "What is 15% of 200?")
        print_result("Answer", answer)

        # Check if answer contains expected result
        if "30" in answer:
            print_success("Calculator tool executed successfully!")
        else:
            print_error(f"Expected '30' in answer, got: {answer}")

    except Exception as e:
        print_error(f"Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()


def test_calculator_complex():
    """Test 2: Complex calculator operations"""
    print_test("Calculator Tool - Complex Math")

    client = OpenAI(api_key=Config.OPENAI_KEY)
    agent = AgentService(openai_client=client, max_iterations=5)

    try:
        answer = agent.run(
            query="Calculate (500 - 200) / 3 and show the result", context={}
        )

        print_result("Query", "Calculate (500 - 200) / 3")
        print_result("Answer", answer)

        # 300/3 = 100
        if "100" in answer:
            print_success("Complex calculation successful!")
        else:
            print_error(f"Expected '100' in answer, got: {answer}")

    except Exception as e:
        print_error(f"Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()


def test_document_search():
    """Test 3: Document search tool"""
    print_test("Document Search Tool - RAG Integration")

    client = OpenAI(api_key=Config.OPENAI_KEY)
    agent = AgentService(openai_client=client, max_iterations=5)

    try:
        # Get ChromaDB collection
        collection = get_collection()

        # Count documents in collection
        doc_count = collection.count()
        print_result("Documents in collection", doc_count)

        if doc_count == 0:
            print_error(
                "No documents in collection. Please ingest some documents first."
            )
            print(
                f"{Colors.WARNING}Hint: Upload and ingest documents via the web UI first{Colors.ENDC}"
            )
            return

        # Test document search
        answer = agent.run(
            query="Search the documents and tell me who the man ove is",
            context={
                "collection": collection,
                "dept_id": "test_dept",
                "user_id": "test_user",
                "request": MockRequest(),
                "use_hybrid": Config.USE_HYBRID,
                "use_reranker": Config.USE_RERANKER,
            },
        )

        print_result("Query", "Search the documents...")
        print_result("Answer", answer)

        # Check if search was performed
        if (
            "source" in answer.lower()
            or "document" in answer.lower()
            or len(answer) > 50
        ):
            print_success("Document search executed successfully!")
        else:
            print_error(f"Search may have failed or returned no results")

    except Exception as e:
        print_error(f"Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()


def test_multi_step():
    """Test 4: Multi-step reasoning with multiple tool calls"""
    print_test("Multi-Step Reasoning - Chained Tools")

    client = OpenAI(api_key=Config.OPENAI_KEY)
    agent = AgentService(openai_client=client, max_iterations=5)

    try:
        # This should trigger both search AND calculator
        collection = get_collection()

        if collection.count() == 0:
            print_error("No documents in collection. Skipping multi-step test.")
            return

        answer = agent.run(
            query="Search for any numerical data in the document the man called ove, then calculate what a 20% increase would be",
            context={
                "collection": collection,
                "dept_id": "test_dept",
                "user_id": "test_user",
                "request": MockRequest(),
                "use_hybrid": Config.USE_HYBRID,
                "use_reranker": Config.USE_RERANKER,
            },
        )

        print_result("Query", "Search for numerical data + calculate 20% increase")
        print_result("Answer", answer)

        # Check if multiple tools were used
        if len(answer) > 100:  # Multi-step answers are typically longer
            print_success("Multi-step reasoning appears to have worked!")
        else:
            print_error("Answer seems too short for multi-step reasoning")

    except Exception as e:
        print_error(f"Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()


def test_no_tool_needed():
    """Test 5: Simple question that doesn't need tools"""
    print_test("No Tool Needed - Direct Answer")

    client = OpenAI(api_key=Config.OPENAI_KEY)
    agent = AgentService(openai_client=client, max_iterations=5)

    try:
        answer = agent.run(query="What is the capital of France?", context={})

        print_result("Query", "What is the capital of France?")
        print_result("Answer", answer)

        if "Paris" in answer:
            print_success("Agent answered directly without tools!")
        else:
            print_error(f"Expected 'Paris' in answer, got: {answer}")

    except Exception as e:
        print_error(f"Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()


def test_max_iterations():
    """Test 6: Max iterations safety check"""
    print_test("Max Iterations - Safety Check")

    client = OpenAI(api_key=Config.OPENAI_KEY)

    # Create agent with very low max_iterations for testing
    agent = AgentService(openai_client=client, max_iterations=1)

    try:
        # This query might require multiple tool calls
        answer = agent.run(
            query="Calculate 10 * 20, then calculate 50% of that result, then add 25",
            context={},
        )

        print_result("Query", "Multi-step calculation (with max_iterations=1)")
        print_result("Answer", answer)

        # Should hit max iterations or succeed quickly
        if "Error: Maximum iterations" in answer or "100" in answer or "125" in answer:
            print_success("Max iterations check working!")
        else:
            print_error(f"Unexpected result: {answer}")

    except Exception as e:
        print_error(f"Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()


def main():
    """Run all tests"""
    print_header("AGENT SERVICE TEST SUITE")

    # Check if OpenAI API key is configured
    if not Config.OPENAI_KEY:
        print_error("OPENAI_API_KEY not configured in .env file!")
        print("Please add your OpenAI API key to backend/.env")
        return

    print_success(f"OpenAI API Key: {Config.OPENAI_KEY[:20]}...")
    print_success(f"Using model: gpt-4o-mini")
    print_success(f"Hybrid search: {Config.USE_HYBRID}")
    print_success(f"Reranker: {Config.USE_RERANKER}")

    # Run tests
    tests = [
        ("Calculator - Basic", test_calculator),
        ("Calculator - Complex", test_calculator_complex),
        ("Document Search", test_document_search),
        ("Multi-Step Reasoning", test_multi_step),
        ("No Tool Needed", test_no_tool_needed),
        ("Max Iterations Safety", test_max_iterations),
    ]

    print(f"\n{Colors.BOLD}Running {len(tests)} tests...{Colors.ENDC}\n")

    for i, (name, test_func) in enumerate(tests, 1):
        print(f"\n{Colors.BOLD}[{i}/{len(tests)}]{Colors.ENDC}")
        test_func()

    # Summary
    print_header("TEST SUITE COMPLETE")
    print(f"{Colors.OKGREEN}All tests executed!{Colors.ENDC}")
    print(
        f"\n{Colors.WARNING}Note: Review the output above to verify results{Colors.ENDC}"
    )
    print(
        f"{Colors.WARNING}Some tests may fail if no documents are ingested{Colors.ENDC}\n"
    )


if __name__ == "__main__":
    main()
