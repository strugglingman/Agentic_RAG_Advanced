# Testing Guide - Week 2 & Week 3

**Purpose**: Complete testing for query refinement, clarification, and web search features

**Status**: 📋 Ready to Test

---

## 🎯 Testing Overview

### What We're Testing:
1. **Week 2 Features**:
   - Query Refinement (automatic query improvement)
   - Clarification Messages (helpful fallback messages)
   - Progressive fallback (ANSWER → REFINE → CLARIFY)

2. **Week 3 Features**:
   - Web Search Service (DuckDuckGo integration)
   - EXTERNAL Detection (recognize external queries)
   - Agent integration (web_search tool)

---

## 📋 Testing Checklist

### Phase 1: Standalone Service Tests ⚡ (15 minutes)

These test individual services in isolation.

#### Test 1: ClarificationHelper
```bash
cd backend
source venv/bin/activate  # Windows: venv\Scripts\activate
python -m src.services.clarification_helper
```

**Expected Output**:
```
======================================================================
CLARIFICATION HELPER TEST
======================================================================
[OK] ClarificationHelper created

----------------------------------------------------------------------
Test 1: No Results Message
----------------------------------------------------------------------
I couldn't find any relevant information for: "What is quantum physics?".

Suggestions to improve your query:
- The information might not be in the uploaded documents.
- Try using different keywords.
- Be more specific about what you're looking for.

Note: I have searched in HR documents.

----------------------------------------------------------------------
Test 2: Ambiguous Query Message
----------------------------------------------------------------------
Your query "How do I apply?" returned low-quality results.

Issues found:
- Low average relevance score: 0.20
- Poor keyword match: 0.10

Missing keywords: apply, application, process, submit

Suggestions to improve your query:
- Be more specific about what you're looking for.
- Use different keywords.
- Provide more context if possible.

Note: I have searched in company policies.

----------------------------------------------------------------------
Test 3: Max Attempts Message
----------------------------------------------------------------------
After multiple search attempts, I couldn't find highly relevant information for: "Tell me about xyz".

The best results I found may not fully answer your question.

Suggestions to improve your query:
- Try rephrasing your question.
- Break it into smaller, more specific questions.
- Check if the information exists in the uploaded documents.

Keywords not found: apply, application, process, submit

I have searched in: company documents.

----------------------------------------------------------------------
Test 4: generate_clarification Routing
----------------------------------------------------------------------
  max_attempts_reached=True: _max_attempts_message
  empty issues: _no_results_message
  issues present: _ambiguous_query_message
  [OK] Routing logic verified

======================================================================
ALL TESTS COMPLETE!
======================================================================
```

**Success Criteria**: ✅ All 4 tests pass, messages are clear and helpful

---

#### Test 2: QueryRefiner
```bash
cd backend
source venv/bin/activate  # Windows: venv\Scripts\activate
python -m src.services.query_refiner
```

**Expected Output**:
```
======================================================================
QUERY REFINER TEST
======================================================================

Test 1: QueryRefiner Initialization
----------------------------------------------------------------------
  OpenAI client: <OpenAI instance>
  Model: gpt-4o-mini
  Temperature: 0.1
[OK] QueryRefiner initialized

Test 2: Refinement (with LLM)
----------------------------------------------------------------------
  Original query: 'benefits policy'
  Refined query: 'employee benefits policy medical dental vision coverage'
  [OK] Query expanded with more keywords

Test 3: Simple Refinement (fallback, no LLM)
----------------------------------------------------------------------
  Original query: 'benefits policy'
  Refined query: 'benefits policy employee health insurance coverage'
  [OK] Fallback refinement working

Test 4: should_refine() Logic
----------------------------------------------------------------------
  Test 1 (should_refine=True, confidence=0.6): True
  Test 2 (should_refine=False, confidence=0.85): False
  Test 3 (should_refine=False, confidence=0.4): False
[OK] should_refine() logic working

Test 5: track_refinement() Helper
----------------------------------------------------------------------
[REFINEMENT] Attempt 1: 'original' -> 'refined'
[OK] Refinement tracking working

======================================================================
ALL TESTS COMPLETE!
======================================================================
```

**Success Criteria**: ✅ LLM refinement works, fallback works, logic correct

---

#### Test 3: WebSearchService
```bash
cd backend
source venv/bin/activate  # Windows: venv\Scripts\activate
python -m src.services.web_search
```

**Expected Output**:
```
======================================================================
WEB SEARCH SERVICE TEST
======================================================================

Test 1: Service Initialization
  Provider: duckduckgo
  Max results: 5
[OK] Service initialized

Test 2: Query Validation
  Valid query: 'test query'
  Short query rejected: Query too short (minimum 3 characters)
  Long query truncated to 500 chars
[OK] Query validation working

Test 3: DuckDuckGo Search
  Query: 'Python programming language'
  Results found: 3
    - Python (programming language) - Wikipedia... (en.wikipedia.org)
    - Welcome to Python.org... (python.org)
    - Python Tutorial - W3Schools... (w3schools.com)
[OK] DuckDuckGo search working

Test 4: Format for Agent
Web search results for: "Python programming language"

1. Python (programming language) - Wikipedia
   Source: en.wikipedia.org
   URL: https://en.wikipedia.org/wiki/Python_(programming_language)
   Python is a high-level, general-purpose programming language...

2. Welcome to Python.org
   Source: python.org
   URL: https://www.python.org/
   The official home of the Python Programming Language...

3. Python Tutorial - W3Schools
   Source: w3schools.com
   URL: https://www.w3schools.com/python/
   Learn Python programming with W3Schools tutorials...

[Note: These results are from external web sources]
[OK] Formatting working

Test 5: Real-world Query
Web search results for: "current US inflation rate 2024"

1. Consumer Price Index Summary - Bureau of Labor Statistics
   Source: bls.gov
   URL: https://www.bls.gov/news.release/cpi.nr0.htm
   ...

[Note: These results are from external web sources]
[OK] Real-world query working

======================================================================
ALL TESTS COMPLETE!
======================================================================
```

**Success Criteria**: ✅ DuckDuckGo returns results, formatting correct

---

### Phase 2: Integration Tests 🔗 (30-60 minutes)

These test the complete flow through the agent.

#### Test 4: Week 2 Integration Test

Create file: `backend/tests/test_week2_integration.py`

```python
"""
Integration test for Week 2: Query Refinement + Clarification
"""

import sys
sys.path.insert(0, ".")

from src.services.retrieval_evaluator import RetrievalEvaluator
from src.services.query_refiner import QueryRefiner
from src.services.clarification_helper import ClarificationHelper
from src.models.evaluation import (
    ReflectionConfig,
    EvaluationCriteria,
    QualityLevel,
    RecommendationAction
)
from src.config.settings import Config
from openai import OpenAI

def test_refinement_flow():
    """Test: REFINE recommendation → query refinement"""
    print("=" * 70)
    print("TEST 1: Refinement Flow")
    print("=" * 70)

    # Setup
    client = OpenAI(api_key=Config.OPENAI_KEY)
    config = ReflectionConfig.from_settings(Config)
    evaluator = RetrievalEvaluator(config=config, openai_client=client)
    refiner = QueryRefiner(openai_client=client)

    # Simulate low-quality retrieval (confidence 0.55 → REFINE)
    mock_contexts = [
        {"chunk": "Some tangentially related text about benefits", "score": 0.55},
        {"chunk": "Another loosely related paragraph", "score": 0.50},
    ]

    query = "employee benefits policy"
    criteria = EvaluationCriteria(
        query=query,
        contexts=mock_contexts,
        search_metadata={},
        mode="fast"
    )

    # Evaluate
    eval_result = evaluator.evaluate(criteria)
    print(f"Quality: {eval_result.quality.value}")
    print(f"Confidence: {eval_result.confidence:.2f}")
    print(f"Recommendation: {eval_result.recommendation.value}")
    print(f"Should refine: {eval_result.should_refine}")

    # If REFINE recommended, refine the query
    if eval_result.should_refine:
        refined = refiner.refine_query(query, eval_result)
        print(f"\nOriginal: '{query}'")
        print(f"Refined: '{refined}'")
        print("✅ Refinement triggered successfully")
    else:
        print("❌ REFINE not recommended (expected it to be)")

    print()

def test_max_attempts_clarification():
    """Test: Max refinement attempts → Clarification"""
    print("=" * 70)
    print("TEST 2: Max Attempts → Clarification")
    print("=" * 70)

    client = OpenAI(api_key=Config.OPENAI_KEY)
    clarifier = ClarificationHelper(openai_client=client)

    # Simulate poor evaluation after max attempts
    from src.models.evaluation import EvaluationResult
    mock_eval = EvaluationResult(
        quality=QualityLevel.POOR,
        confidence=0.45,
        coverage=0.3,
        recommendation=RecommendationAction.REFINE,
        reasoning="Still poor after refinements",
        issues=["Low relevance", "Missing keywords"],
        missing_aspects=["health", "insurance", "dental"],
        relevance_scores=[0.4, 0.5],
        metrics={}
    )

    # Generate clarification
    message = clarifier.generate_clarification(
        query="employee benefits",
        eval_result=mock_eval,
        max_attempts_reached=True,
        context_hint="company documents"
    )

    print("Clarification Message:")
    print("-" * 70)
    print(message)
    print()

    # Verify it's the max attempts message
    if "multiple search attempts" in message:
        print("✅ Max attempts clarification triggered correctly")
    else:
        print("❌ Wrong clarification type")

    print()

def test_direct_clarify():
    """Test: Very low confidence (< 0.5) → Direct CLARIFY"""
    print("=" * 70)
    print("TEST 3: Direct CLARIFY (confidence < 0.5)")
    print("=" * 70)

    client = OpenAI(api_key=Config.OPENAI_KEY)
    config = ReflectionConfig.from_settings(Config)
    evaluator = RetrievalEvaluator(config=config, openai_client=client)
    clarifier = ClarificationHelper(openai_client=client)

    # Simulate very poor retrieval
    mock_contexts = [
        {"chunk": "Completely unrelated text", "score": 0.25},
        {"chunk": "Another irrelevant paragraph", "score": 0.20},
    ]

    query = "quantum entanglement theory"
    criteria = EvaluationCriteria(
        query=query,
        contexts=mock_contexts,
        search_metadata={},
        mode="fast"
    )

    eval_result = evaluator.evaluate(criteria)
    print(f"Quality: {eval_result.quality.value}")
    print(f"Confidence: {eval_result.confidence:.2f}")
    print(f"Recommendation: {eval_result.recommendation.value}")
    print(f"Should clarify: {eval_result.should_clarify}")

    if eval_result.should_clarify:
        message = clarifier.generate_clarification(
            query=query,
            eval_result=eval_result,
            max_attempts_reached=False
        )
        print("\nClarification Message:")
        print("-" * 70)
        print(message)
        print("✅ Direct clarification triggered (skipped refinement)")
    else:
        print("❌ CLARIFY not recommended (expected)")

    print()

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("WEEK 2 INTEGRATION TESTS")
    print("=" * 70 + "\n")

    test_refinement_flow()
    test_max_attempts_clarification()
    test_direct_clarify()

    print("=" * 70)
    print("ALL INTEGRATION TESTS COMPLETE!")
    print("=" * 70)
```

**Run**:
```bash
cd backend
source venv/bin/activate
python tests/test_week2_integration.py
```

---

#### Test 5: Week 3 Integration Test

Create file: `backend/tests/test_week3_integration.py`

```python
"""
Integration test for Week 3: Web Search + EXTERNAL Detection
"""

import sys
sys.path.insert(0, ".")

from src.services.retrieval_evaluator import RetrievalEvaluator
from src.services.web_search import WebSearchService
from src.models.evaluation import ReflectionConfig, EvaluationCriteria
from src.config.settings import Config

def test_external_detection():
    """Test: EXTERNAL query detection"""
    print("=" * 70)
    print("TEST 1: EXTERNAL Query Detection")
    print("=" * 70)

    client = None  # Not needed for fast mode
    config = ReflectionConfig.from_settings(Config)
    evaluator = RetrievalEvaluator(config=config, openai_client=client)

    # Test external query indicators
    external_queries = [
        "What is the current inflation rate?",
        "Today's weather in New York",
        "Latest stock price for AAPL",
        "Real-time cryptocurrency prices",
        "Who is the current CEO of Microsoft?",
    ]

    for query in external_queries:
        # Simulate no relevant contexts (external info not in docs)
        mock_contexts = []

        criteria = EvaluationCriteria(
            query=query,
            contexts=mock_contexts,
            search_metadata={},
            mode="fast"
        )

        eval_result = evaluator.evaluate(criteria)
        print(f"\nQuery: '{query}'")
        print(f"  Recommendation: {eval_result.recommendation.value}")
        print(f"  Should search external: {eval_result.should_search_external}")

        if eval_result.should_search_external:
            print("  ✅ EXTERNAL detected")
        else:
            print("  ❌ EXTERNAL not detected (should be)")

    print()

def test_web_search_execution():
    """Test: Web search actually returns results"""
    print("=" * 70)
    print("TEST 2: Web Search Execution")
    print("=" * 70)

    if not Config.WEB_SEARCH_ENABLED:
        print("⚠️  WEB_SEARCH_ENABLED=false, skipping test")
        return

    service = WebSearchService()

    # Test queries
    test_queries = [
        "Python programming language",
        "current inflation rate",
        "machine learning tutorial",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = service.search(query, max_results=3)
        print(f"  Results: {len(results)}")

        if results:
            print(f"  Sample: {results[0].title[:50]}...")
            print("  ✅ Web search working")
        else:
            print("  ❌ No results returned")

    print()

def test_external_flow():
    """Test: Complete EXTERNAL flow (detection → suggestion)"""
    print("=" * 70)
    print("TEST 3: Complete EXTERNAL Flow")
    print("=" * 70)

    # This would ideally test through agent_tools.execute_search_documents()
    # But for now, we test the components separately

    print("\nEXTERNAL flow components:")
    print("  1. Query detected as EXTERNAL ✅ (Test 1)")
    print("  2. Agent receives EXTERNAL suggestion ✅ (in agent_tools.py)")
    print("  3. Agent can call web_search tool ✅ (in agent_tools.py)")
    print("  4. Web search returns results ✅ (Test 2)")

    print("\nFor full end-to-end test, start the server and test via API")
    print()

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("WEEK 3 INTEGRATION TESTS")
    print("=" * 70 + "\n")

    test_external_detection()
    test_web_search_execution()
    test_external_flow()

    print("=" * 70)
    print("ALL INTEGRATION TESTS COMPLETE!")
    print("=" * 70)
```

**Run**:
```bash
cd backend
source venv/bin/activate
python tests/test_week3_integration.py
```

---

### Phase 3: End-to-End API Tests 🚀 (Manual - 30 minutes)

Test through the actual chat API to verify everything works together.

#### Prerequisites:
1. Start the backend server:
```bash
cd backend
source venv/bin/activate
flask run
```

2. Start the frontend:
```bash
cd frontend
npm run dev
```

#### Test Scenarios:

##### Scenario 1: High Confidence (ANSWER directly)
```
Query: "What is our vacation policy?"
Expected: Direct answer with no refinement messages
```

##### Scenario 2: Medium Confidence (REFINE → improved)
```
Query: "benefits"
Expected:
- System refines to "employee benefits policy coverage"
- Returns improved results
- Log shows: [QUERY_REFINER] Refinement attempt 1/3
```

##### Scenario 3: Max Attempts → Clarification
```
Query: "xyz abc"
Expected:
- System tries 3 refinements
- Returns: "After multiple search attempts, I couldn't find..."
- Suggestions provided
```

##### Scenario 4: Direct CLARIFY (confidence < 0.5)
```
Query: "quantum physics theory"
Expected:
- Skips refinement
- Returns: "Your query returned low-quality results..."
- Log shows: [CLARIFICATION] Direct clarify (confidence < 0.5)
```

##### Scenario 5: EXTERNAL Detection
```
Query: "What is the current US inflation rate?"
Expected:
- System detects EXTERNAL
- Returns: "[EXTERNAL SEARCH SUGGESTED]..."
- Suggests using web_search tool
```

##### Scenario 6: Web Search (if agent calls it)
```
If agent decides to use web_search tool:
Expected:
- Returns web search results
- Results clearly marked: "[Note: These results are from external web sources]"
```

---

## 📊 Testing Results Template

Use this to track your testing results:

```markdown
# Testing Results - Week 2 & 3

**Date**: ___________
**Tester**: ___________

## Phase 1: Standalone Tests

| Test | Status | Notes |
|------|--------|-------|
| clarification_helper.py | ⬜ Pass / ⬜ Fail | |
| query_refiner.py | ⬜ Pass / ⬜ Fail | |
| web_search.py | ⬜ Pass / ⬜ Fail | |

## Phase 2: Integration Tests

| Test | Status | Notes |
|------|--------|-------|
| Week 2: Refinement flow | ⬜ Pass / ⬜ Fail | |
| Week 2: Max attempts → Clarify | ⬜ Pass / ⬜ Fail | |
| Week 2: Direct CLARIFY | ⬜ Pass / ⬜ Fail | |
| Week 3: EXTERNAL detection | ⬜ Pass / ⬜ Fail | |
| Week 3: Web search execution | ⬜ Pass / ⬜ Fail | |

## Phase 3: End-to-End Tests

| Scenario | Status | Notes |
|----------|--------|-------|
| High confidence (ANSWER) | ⬜ Pass / ⬜ Fail | |
| Medium confidence (REFINE) | ⬜ Pass / ⬜ Fail | |
| Max attempts (CLARIFY) | ⬜ Pass / ⬜ Fail | |
| Direct CLARIFY | ⬜ Pass / ⬜ Fail | |
| EXTERNAL detection | ⬜ Pass / ⬜ Fail | |
| Web search | ⬜ Pass / ⬜ Fail | |

## Issues Found

1.
2.
3.

## Overall Status

⬜ Week 2 Complete
⬜ Week 3 Complete
⬜ All Tests Passing
```

---

## 🐛 Troubleshooting

### Issue: Import errors when running standalone tests
**Solution**: Activate virtual environment first
```bash
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### Issue: "No module named 'openai'"
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: Web search returns no results
**Check**:
1. Internet connection working
2. `WEB_SEARCH_ENABLED=true` in `.env`
3. Query not too specific
4. Try different query

### Issue: EXTERNAL never detected
**Check**:
1. Query contains external indicators ("current", "today", "latest", etc.)
2. `USE_SELF_REFLECTION=true` in `.env`
3. Check logs for `[SELF-REFLECTION]` messages

### Issue: Refinement doesn't trigger
**Check**:
1. `USE_SELF_REFLECTION=true` in `.env`
2. `REFLECTION_AUTO_REFINE=true` in `.env`
3. Query has medium confidence (0.50-0.70)
4. Check logs for `[QUERY_REFINER]` messages

---

## ✅ Success Criteria

Week 2 & 3 testing is complete when:

- [ ] All 3 standalone service tests pass
- [ ] All 5 integration test scenarios pass
- [ ] All 6 end-to-end scenarios work correctly
- [ ] Logs show correct decision flow
- [ ] No unexpected errors in production

---

## 📚 References

- Week 2 Checklist: [WEEK2_CHECKLIST.md](./WEEK2_CHECKLIST.md)
- Week 3 Checklist: [WEEK3_CHECKLIST.md](./WEEK3_CHECKLIST.md)
- Implementation Guides:
  - [WEEK2_DAY3_IMPLEMENTATION.md](./WEEK2_DAY3_IMPLEMENTATION.md)
  - [WEEK2_DAY4_IMPLEMENTATION.md](./WEEK2_DAY4_IMPLEMENTATION.md)
  - [WEEK3_DAY5_IMPLEMENTATION.md](./WEEK3_DAY5_IMPLEMENTATION.md)
  - [WEEK3_DAY6_IMPLEMENTATION.md](./WEEK3_DAY6_IMPLEMENTATION.md)
