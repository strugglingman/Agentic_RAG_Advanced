# Week 2 - Day 3: Query Refinement Implementation

**Focus**: Core query refinement logic with loop prevention
**Time**: 4-5 hours
**Prerequisites**: Week 1 Complete ✅

---

## 📋 Overview

Implement the core query refinement system that:
1. Detects when refinement is needed (REFINE recommendation)
2. Reformulates the query using LLM
3. Retries the search with refined query
4. Tracks attempts to prevent infinite loops

---

## 🎯 What We're Building

### Before (Week 1):
```python
# agent_tools.py - execute_search_documents()
ctx = retrieve(query, ...)
eval_result = evaluator.evaluate(...)
print(f"[REFLECTION] Recommendation: {eval_result.recommendation.value}")
return format_contexts(ctx)  # ← Always returns contexts, even if poor
```

### After (Week 2 Day 3):
```python
# agent_tools.py - execute_search_documents()
ctx = retrieve(query, ...)
eval_result = evaluator.evaluate(...)

# NEW: Take action on recommendation
if eval_result.should_refine and can_refine(context):
    refined_query = refiner.refine_query(query, eval_result)
    ctx = retrieve(refined_query, ...)  # ← Retry with better query
    eval_result = evaluator.evaluate(...)  # ← Re-evaluate

return format_contexts(ctx)
```

---

## 🏗️ Part 1: Create QueryRefiner Service

### Step 1.1: Create the QueryRefiner class

**File**: `backend/src/services/query_refiner.py`

```python
"""
Query refinement service for self-reflection system.

This service reformulates user queries when retrieval quality is poor,
using LLM and evaluation feedback to create better search queries.
"""

from typing import Optional, List
from openai import OpenAI
from src.models.evaluation import EvaluationResult


class QueryRefiner:
    """
    Refines user queries based on evaluation feedback.

    Uses LLM to reformulate queries when retrieval evaluation
    indicates poor quality (REFINE recommendation).
    """

    def __init__(
        self,
        openai_client: OpenAI,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
    ):
        """
        Initialize the query refiner.

        Args:
            openai_client: OpenAI client for LLM calls
            model: Model to use for refinement (default: gpt-4o-mini)
            temperature: Temperature for creativity (0.3 = balanced)
        """
        self.client = openai_client
        self.model = model
        self.temperature = temperature

    def refine_query(
        self,
        original_query: str,
        eval_result: EvaluationResult,
        context_hint: Optional[str] = None,
    ) -> str:
        """
        Refine a query based on evaluation feedback.

        Args:
            original_query: The original user query
            eval_result: Evaluation result with issues/missing_aspects
            context_hint: Optional context about the document collection

        Returns:
            Refined query string

        Example:
            original: "Tell me about PTO"
            refined: "employee paid time off vacation policy accrual"
        """
        # Build refinement prompt
        prompt = self._build_refinement_prompt(
            original_query, eval_result, context_hint
        )

        # Call LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=100,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a query refinement expert. "
                            "Your job is to reformulate user queries to improve "
                            "document search results. Make queries more specific, "
                            "expand abbreviations, and add relevant keywords."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            refined = response.choices[0].message.content.strip()

            # Clean up (remove quotes, extra whitespace)
            refined = refined.strip('"\'').strip()

            # Fallback: if refinement looks bad, return original
            if len(refined) < 3 or refined == original_query:
                return original_query

            return refined

        except Exception as e:
            print(f"[QUERY_REFINER] LLM call failed: {e}")
            # Fallback: expand abbreviations manually
            return self._simple_refinement(original_query, eval_result)

    def _build_refinement_prompt(
        self,
        original_query: str,
        eval_result: EvaluationResult,
        context_hint: Optional[str] = None,
    ) -> str:
        """Build the LLM prompt for query refinement."""

        prompt_parts = [
            f"Original query: \"{original_query}\"",
            "",
            "The search returned poor results. Here's why:",
        ]

        # Add issues from evaluation
        if eval_result.issues:
            prompt_parts.append("Issues:")
            for issue in eval_result.issues:
                prompt_parts.append(f"  - {issue}")
            prompt_parts.append("")

        # Add missing aspects
        if eval_result.missing_aspects:
            prompt_parts.append("Missing keywords in results:")
            prompt_parts.append(f"  {', '.join(eval_result.missing_aspects)}")
            prompt_parts.append("")

        # Add context hint if provided
        if context_hint:
            prompt_parts.append(f"Context: {context_hint}")
            prompt_parts.append("")

        # Instructions
        prompt_parts.extend([
            "Task: Reformulate the query to get better search results.",
            "",
            "Guidelines:",
            "1. Expand abbreviations (e.g., 'PTO' → 'paid time off vacation')",
            "2. Add relevant keywords from missing aspects",
            "3. Make the query more specific and searchable",
            "4. Keep it concise (under 15 words)",
            "5. Focus on document search, not conversational",
            "",
            "Return ONLY the refined query, nothing else.",
        ])

        return "\n".join(prompt_parts)

    def _simple_refinement(
        self,
        original_query: str,
        eval_result: EvaluationResult,
    ) -> str:
        """
        Simple fallback refinement without LLM.

        Expands common abbreviations and adds missing keywords.
        """
        refined = original_query.lower()

        # Common abbreviations expansion
        abbreviations = {
            "pto": "paid time off vacation",
            "hr": "human resources",
            "401k": "retirement savings plan",
            "q1": "first quarter",
            "q2": "second quarter",
            "q3": "third quarter",
            "q4": "fourth quarter",
            "yoy": "year over year",
            "roi": "return on investment",
        }

        for abbr, expansion in abbreviations.items():
            if abbr in refined.split():
                refined = refined.replace(abbr, expansion)

        # Add missing aspects as keywords
        if eval_result.missing_aspects:
            # Add top 3 missing keywords
            missing_keywords = eval_result.missing_aspects[:3]
            refined = f"{refined} {' '.join(missing_keywords)}"

        return refined.strip()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def should_refine(eval_result: EvaluationResult, context: dict) -> bool:
    """
    Check if refinement should be attempted.

    Args:
        eval_result: Evaluation result
        context: Agent context with refinement tracking

    Returns:
        True if should refine, False otherwise
    """
    # Check if recommendation is REFINE
    if not eval_result.should_refine:
        return False

    # Check refinement attempts
    refinement_count = context.get("_refinement_count", 0)
    max_attempts = context.get("_max_refinement_attempts", 3)

    if refinement_count >= max_attempts:
        print(f"[QUERY_REFINER] Max refinement attempts ({max_attempts}) reached")
        return False

    return True


def track_refinement(context: dict, original_query: str, refined_query: str):
    """
    Track refinement in context for loop prevention.

    Args:
        context: Agent context dict (modified in-place)
        original_query: Original query
        refined_query: Refined query
    """
    # Initialize tracking if first refinement
    if "_refinement_count" not in context:
        context["_refinement_count"] = 0
        context["_original_query"] = original_query
        context["_refinement_history"] = []

    # Increment count
    context["_refinement_count"] += 1

    # Add to history
    context["_refinement_history"].append({
        "attempt": context["_refinement_count"],
        "from": original_query,
        "to": refined_query,
    })

    print(f"[QUERY_REFINER] Refinement attempt {context['_refinement_count']}")
    print(f"[QUERY_REFINER] Original: {original_query}")
    print(f"[QUERY_REFINER] Refined: {refined_query}")


# =============================================================================
# TESTING (run with: python -m src.services.query_refiner)
# =============================================================================

if __name__ == "__main__":
    from openai import OpenAI
    from src.config.settings import Config
    from src.models.evaluation import RecommendationAction, QualityLevel

    print("=" * 70)
    print("QUERY REFINER TEST")
    print("=" * 70)

    # Create refiner
    client = OpenAI(api_key=Config.OPENAI_KEY)
    refiner = QueryRefiner(openai_client=client)

    # Test 1: Abbreviation expansion
    print("\nTest 1: Abbreviation Expansion")
    print("-" * 70)

    original = "Tell me about PTO"
    mock_eval = EvaluationResult(
        quality=QualityLevel.PARTIAL,
        confidence=0.35,
        coverage=0.2,
        recommendation=RecommendationAction.REFINE,
        reasoning="Poor keyword match",
        issues=["Poor keyword match: 0.20"],
        missing_aspects=["vacation", "time", "off", "policy"],
        relevance_scores=[0.3, 0.4],
        metrics={},
    )

    refined = refiner.refine_query(original, mock_eval)
    print(f"Original: {original}")
    print(f"Refined: {refined}")

    # Test 2: Ambiguous query
    print("\nTest 2: Ambiguous Query")
    print("-" * 70)

    original = "How do I apply?"
    mock_eval = EvaluationResult(
        quality=QualityLevel.POOR,
        confidence=0.15,
        coverage=0.0,
        recommendation=RecommendationAction.REFINE,
        reasoning="No contexts retrieved",
        issues=["No contexts retrieved"],
        missing_aspects=["apply", "application", "process"],
        relevance_scores=[],
        metrics={},
    )

    refined = refiner.refine_query(original, mock_eval, context_hint="HR policy documents")
    print(f"Original: {original}")
    print(f"Refined: {refined}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
```

**Action**: Create this file and test it:
```bash
cd backend
python -m src.services.query_refiner
```

---

## 🏗️ Part 2: Add Configuration for Refinement

### Step 2.1: Update settings.py

**File**: `backend/src/config/settings.py`

**Add these settings** (find the self-reflection section):

```python
# Query Refinement Settings (Week 2)
REFLECTION_AUTO_REFINE: bool = os.getenv("REFLECTION_AUTO_REFINE", "true").lower() == "true"
REFLECTION_MAX_REFINEMENT_ATTEMPTS: int = int(os.getenv("REFLECTION_MAX_REFINEMENT_ATTEMPTS", "3"))
```

### Step 2.2: Update .env

**File**: `backend/.env`

**Add these lines** (after existing REFLECTION_* settings):

```bash
# Week 2: Query Refinement
REFLECTION_AUTO_REFINE=true                  # Enable automatic query refinement
REFLECTION_MAX_REFINEMENT_ATTEMPTS=3         # Max refinement attempts before giving up
```

### Step 2.3: Update ReflectionConfig

**File**: `backend/src/models/evaluation.py`

**Find the ReflectionConfig class** and add these fields:

```python
@dataclass
class ReflectionConfig:
    # ... existing fields ...

    # Week 2: Query Refinement settings
    auto_refine: bool = True
    max_refinement_attempts: int = 3
```

**Update the `from_settings()` method**:

```python
@classmethod
def from_settings(cls, config):
    """Load configuration from settings."""
    return cls(
        # ... existing fields ...

        # Week 2: Refinement settings
        auto_refine=config.REFLECTION_AUTO_REFINE,
        max_refinement_attempts=config.REFLECTION_MAX_REFINEMENT_ATTEMPTS,
    )
```

---

## 🏗️ Part 3: Integrate Refinement into agent_tools.py

### Step 3.1: Import the refiner

**File**: `backend/src/services/agent_tools.py`

**Add imports at the top**:

```python
from src.services.query_refiner import QueryRefiner, should_refine, track_refinement
```

### Step 3.2: Modify execute_search_documents()

**Find the function `execute_search_documents()`** and modify the section after evaluation:

**Current code** (after line ~150):
```python
        # Continue with normal flow: format and return contexts
        for c in ctx:
            c["chunk"] = scrub_context(c.get("chunk", ""))
```

**Replace with**:
```python
        # Week 2: Check if refinement is needed
        if Config.USE_SELF_REFLECTION and eval_result.should_refine:
            if should_refine(eval_result, context):
                # Initialize refiner
                refiner = QueryRefiner(openai_client=context.get("openai_client"))

                # Refine query
                refined_query = refiner.refine_query(query, eval_result)

                # Track refinement
                track_refinement(context, query, refined_query)

                # Retry search with refined query
                print(f"[QUERY_REFINER] Retrying search with refined query...")
                ctx = retrieve(
                    query=refined_query,
                    collection_name=collection_name,
                    top_k=top_k,
                    use_hybrid=use_hybrid,
                    use_reranker=use_reranker,
                    where=where,
                )

                # Store refined contexts
                context["_retrieved_contexts"].extend(ctx)

                # Re-evaluate refined results
                criteria = EvaluationCriteria(
                    query=refined_query,
                    contexts=ctx,
                    search_metadata={
                        "hybrid": use_hybrid,
                        "reranker": use_reranker,
                        "top_k": top_k,
                        "refinement_attempt": context.get("_refinement_count", 0),
                    },
                    mode=config.mode,
                )

                eval_result = evaluator.evaluate(criteria)
                context["_last_evaluation"] = eval_result

                # Log refined evaluation
                print(f"[SELF-REFLECTION] (After Refinement) Quality: {eval_result.quality.value}, "
                      f"Confidence: {eval_result.confidence:.2f}, "
                      f"Recommendation: {eval_result.recommendation.value}")

        # Continue with normal flow: format and return contexts
        for c in ctx:
            c["chunk"] = scrub_context(c.get("chunk", ""))
```

### Step 3.3: Initialize refinement tracking

**At the start of `execute_search_documents()`**, after getting parameters, add:

```python
    # Initialize refinement tracking (Week 2)
    if "_refinement_count" not in context:
        context["_refinement_count"] = 0
        context["_max_refinement_attempts"] = Config.REFLECTION_MAX_REFINEMENT_ATTEMPTS
```

---

## 🧪 Part 4: Testing

### Step 4.1: Unit test the refiner

**File**: `backend/tests/test_query_refiner.py`

```python
"""
Unit tests for QueryRefiner.

Run with: pytest tests/test_query_refiner.py -v
"""

import pytest
from src.services.query_refiner import QueryRefiner, should_refine, track_refinement
from src.models.evaluation import (
    EvaluationResult,
    QualityLevel,
    RecommendationAction,
)


def test_simple_refinement():
    """Test simple refinement without LLM."""
    from src.services.query_refiner import QueryRefiner

    refiner = QueryRefiner(openai_client=None)  # No client for simple test

    mock_eval = EvaluationResult(
        quality=QualityLevel.PARTIAL,
        confidence=0.3,
        coverage=0.2,
        recommendation=RecommendationAction.REFINE,
        reasoning="Poor quality",
        issues=["Low keyword overlap"],
        missing_aspects=["vacation", "policy"],
        relevance_scores=[0.3],
        metrics={},
    )

    original = "Tell me about PTO"
    refined = refiner._simple_refinement(original, mock_eval)

    assert "paid time off" in refined.lower()
    assert "vacation" in refined.lower()


def test_should_refine_logic():
    """Test refinement decision logic."""
    mock_eval = EvaluationResult(
        quality=QualityLevel.PARTIAL,
        confidence=0.3,
        coverage=0.2,
        recommendation=RecommendationAction.REFINE,
        reasoning="",
        issues=[],
        missing_aspects=[],
        relevance_scores=[],
        metrics={},
    )

    # Should refine on first attempt
    context = {"_refinement_count": 0, "_max_refinement_attempts": 3}
    assert should_refine(mock_eval, context) == True

    # Should not refine after max attempts
    context = {"_refinement_count": 3, "_max_refinement_attempts": 3}
    assert should_refine(mock_eval, context) == False


def test_track_refinement():
    """Test refinement tracking."""
    context = {}

    track_refinement(context, "original query", "refined query")

    assert context["_refinement_count"] == 1
    assert context["_original_query"] == "original query"
    assert len(context["_refinement_history"]) == 1
    assert context["_refinement_history"][0]["to"] == "refined query"
```

**Run tests**:
```bash
cd backend
pytest tests/test_query_refiner.py -v
```

### Step 4.2: Manual integration test

**Test Scenario**: Poor query that benefits from refinement

1. **Enable self-reflection** in `.env`:
   ```bash
   USE_SELF_REFLECTION=true
   REFLECTION_MODE=fast
   REFLECTION_AUTO_REFINE=true
   ```

2. **Start server**:
   ```bash
   cd backend
   python app.py
   ```

3. **Send test query** (assuming you have HR docs):
   ```bash
   curl -X POST http://localhost:5000/chat/agent \
     -H "Content-Type: application/json" \
     -d '{
       "message": "Tell me about PTO",
       "collection_name": "hr_docs"
     }'
   ```

4. **Check logs** for:
   ```
   [SELF-REFLECTION] Quality: partial, Confidence: 0.35, Recommendation: refine
   [QUERY_REFINER] Refinement attempt 1
   [QUERY_REFINER] Original: Tell me about PTO
   [QUERY_REFINER] Refined: employee paid time off vacation policy
   [QUERY_REFINER] Retrying search with refined query...
   [SELF-REFLECTION] (After Refinement) Quality: good, Confidence: 0.78, Recommendation: answer
   ```

---

## ✅ Day 3 Completion Checklist

- [ ] `query_refiner.py` created and tested
- [ ] QueryRefiner class implements `refine_query()` method
- [ ] Simple fallback refinement works (abbreviation expansion)
- [ ] LLM-based refinement works with mock data
- [ ] Configuration added to settings.py and .env
- [ ] ReflectionConfig updated with refinement fields
- [ ] agent_tools.py modified to use QueryRefiner
- [ ] Refinement tracking prevents infinite loops
- [ ] Unit tests passing
- [ ] Manual integration test shows refinement working
- [ ] Logs clearly show refinement attempts and outcomes

---

## 🎯 Success Criteria

Day 3 is complete when:

✅ Query refinement triggers automatically when eval returns REFINE
✅ Refined queries produce better search results
✅ Refinement tracking prevents infinite loops (max 3 attempts)
✅ Logs show clear refinement flow: original → refined → retry → result
✅ Tests validate both LLM and fallback refinement
✅ Configuration allows enabling/disabling refinement

---

## 🚀 Next Steps

**Tomorrow (Day 4)**:
- Implement clarification logic (CLARIFY recommendation)
- Add progressive fallback (REFINE → REFINE → CLARIFY)
- Comprehensive end-to-end testing
- Documentation updates

---

**Estimated Time**: 4-5 hours
**Difficulty**: Medium (LLM integration + state tracking)
**Dependencies**: Week 1 complete, OpenAI client available
