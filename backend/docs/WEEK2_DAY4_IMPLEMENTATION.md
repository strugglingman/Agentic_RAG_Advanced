# Week 2 - Day 4: Clarification & Progressive Fallback

**Focus**: Clarification logic, progressive fallback, comprehensive testing
**Time**: 3-4 hours
**Prerequisites**: Day 3 Complete (Query refinement working)

---

## 📋 Overview

Implement the remaining Week 2 features:
1. **Clarification logic**: Generate helpful messages when queries are ambiguous
2. **Progressive fallback**: Escalate from REFINE → EXTERNAL/CLARIFY after max attempts
3. **Comprehensive testing**: Validate all refinement scenarios
4. **Documentation**: Update guides and examples

---

## 🎯 What We're Building

### Clarification Flow:
```
User: "How do I apply?"
  ↓
Agent searches → No relevant contexts found
  ↓
Evaluator: Recommendation=CLARIFY
  ↓
Agent: "I couldn't find relevant information. Could you clarify what you'd like to apply for?
        For example: job application, leave request, benefits enrollment, or expense reimbursement."
```

### Progressive Fallback Flow:
```
User: "Tell me about xyz" (very poor query)
  ↓
Attempt 1: REFINE → retry → still poor
  ↓
Attempt 2: REFINE → retry → still poor
  ↓
Attempt 3: REFINE → retry → still poor
  ↓
Max attempts reached → Fallback to CLARIFY
  ↓
Agent: "After multiple refinement attempts, I couldn't find relevant information.
        Could you please rephrase your question or provide more context?"
```

---

## 🏗️ Part 1: Create Clarification Helper

### Step 1.1: Create clarification_helper.py

**File**: `backend/src/services/clarification_helper.py`

```python
"""
Clarification message generation for self-reflection system.

Generates helpful messages when queries are ambiguous or fail to retrieve
relevant information after refinement attempts.
"""

from typing import Optional
from openai import OpenAI
from src.models.evaluation import EvaluationResult


class ClarificationHelper:
    """
    Generates clarification messages for ambiguous queries.

    When retrieval fails or queries are unclear, this class generates
    helpful messages asking the user for more context.
    """

    def __init__(
        self,
        openai_client: Optional[OpenAI] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.5,
    ):
        """
        Initialize clarification helper.

        Args:
            openai_client: Optional OpenAI client for LLM-based suggestions
            model: Model to use (default: gpt-4o-mini)
            temperature: Temperature for creativity (0.5 = balanced)
        """
        self.client = openai_client
        self.model = model
        self.temperature = temperature

    def generate_clarification(
        self,
        query: str,
        eval_result: EvaluationResult,
        refinement_attempted: bool = False,
        context_hint: Optional[str] = None,
    ) -> str:
        """
        Generate a clarification message.

        Args:
            query: Original user query
            eval_result: Evaluation result
            refinement_attempted: Whether refinement was already tried
            context_hint: Optional hint about document collection

        Returns:
            Clarification message string
        """
        # Choose message type based on scenario
        if refinement_attempted:
            return self._max_attempts_message(query, context_hint)
        elif len(eval_result.issues) == 0 or "No contexts" in str(eval_result.issues):
            return self._no_results_message(query, eval_result, context_hint)
        else:
            return self._ambiguous_query_message(query, eval_result, context_hint)

    def _no_results_message(
        self,
        query: str,
        eval_result: EvaluationResult,
        context_hint: Optional[str] = None,
    ) -> str:
        """Generate message when no contexts found."""
        base = f"I couldn't find any relevant information for: \"{query}\""

        suggestions = []

        # Add context-specific suggestions
        if context_hint:
            suggestions.append(f"Note: I searched in {context_hint}")

        # Use LLM to generate suggestions if available
        if self.client:
            llm_suggestions = self._get_llm_suggestions(query, "no_results")
            if llm_suggestions:
                suggestions.extend(llm_suggestions)
        else:
            # Fallback: generic suggestions
            suggestions.append("Could you provide more details or context?")
            suggestions.append("Try rephrasing your question")

        if suggestions:
            return base + "\n\n" + "Suggestions:\n" + "\n".join(f"- {s}" for s in suggestions)
        else:
            return base + "\n\nCould you please rephrase or provide more context?"

    def _ambiguous_query_message(
        self,
        query: str,
        eval_result: EvaluationResult,
        context_hint: Optional[str] = None,
    ) -> str:
        """Generate message for ambiguous queries."""
        base = f"Your query \"{query}\" is unclear or ambiguous."

        # Extract what's missing from evaluation
        issues_text = ""
        if eval_result.issues:
            issues_text = " I found: " + ", ".join(eval_result.issues[:2])

        # Use LLM for suggestions if available
        if self.client:
            llm_suggestions = self._get_llm_suggestions(query, "ambiguous")
            if llm_suggestions:
                suggestions_text = "\n\nDid you mean:\n" + "\n".join(
                    f"- {s}" for s in llm_suggestions
                )
                return base + issues_text + suggestions_text

        # Fallback
        return (
            base + issues_text +
            "\n\nCould you clarify what you're looking for? "
            "For example, please specify the topic, document type, or time period."
        )

    def _max_attempts_message(
        self,
        query: str,
        context_hint: Optional[str] = None,
    ) -> str:
        """Generate message after max refinement attempts."""
        message = (
            f"After multiple attempts, I couldn't find relevant information for: \"{query}\"\n\n"
            "This could mean:\n"
            "- The information isn't in the available documents\n"
            "- The query needs significant clarification\n"
            "- Try searching for a different topic or use different keywords"
        )

        if context_hint:
            message += f"\n\nNote: I searched in {context_hint}"

        return message

    def _get_llm_suggestions(self, query: str, scenario: str) -> list:
        """
        Use LLM to generate contextual suggestions.

        Args:
            query: User query
            scenario: "no_results" or "ambiguous"

        Returns:
            List of suggestion strings (max 3)
        """
        if not self.client:
            return []

        try:
            if scenario == "no_results":
                prompt = (
                    f"The user asked: \"{query}\"\n\n"
                    "No relevant documents were found. Generate 2-3 helpful suggestions "
                    "for what they might be looking for or how to rephrase. "
                    "Format: one suggestion per line, no bullets or numbering."
                )
            else:  # ambiguous
                prompt = (
                    f"The user's query is ambiguous: \"{query}\"\n\n"
                    "Generate 2-3 clarifying questions or interpretations of what they might mean. "
                    "Format: one possibility per line, no bullets or numbering."
                )

            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=150,
                messages=[
                    {
                        "role": "system",
                        "content": "You help users clarify ambiguous search queries.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            content = response.choices[0].message.content.strip()
            suggestions = [
                line.strip().lstrip("-•*").strip()
                for line in content.split("\n")
                if line.strip()
            ]
            return suggestions[:3]  # Max 3 suggestions

        except Exception as e:
            print(f"[CLARIFICATION] LLM call failed: {e}")
            return []


# =============================================================================
# TESTING (run with: python -m src.services.clarification_helper)
# =============================================================================

if __name__ == "__main__":
    from openai import OpenAI
    from src.config.settings import Config
    from src.models.evaluation import RecommendationAction, QualityLevel

    print("=" * 70)
    print("CLARIFICATION HELPER TEST")
    print("=" * 70)

    # Create helper
    client = OpenAI(api_key=Config.OPENAI_KEY)
    helper = ClarificationHelper(openai_client=client)

    # Test 1: No results
    print("\nTest 1: No Results Found")
    print("-" * 70)

    mock_eval = EvaluationResult(
        quality=QualityLevel.POOR,
        confidence=0.0,
        coverage=0.0,
        recommendation=RecommendationAction.CLARIFY,
        reasoning="No contexts retrieved",
        issues=["No contexts retrieved"],
        missing_aspects=["quantum", "physics"],
        relevance_scores=[],
        metrics={},
    )

    message = helper.generate_clarification(
        "What is quantum physics?",
        mock_eval,
        context_hint="HR policy documents"
    )
    print(message)

    # Test 2: Ambiguous query
    print("\n\nTest 2: Ambiguous Query")
    print("-" * 70)

    mock_eval = EvaluationResult(
        quality=QualityLevel.POOR,
        confidence=0.15,
        coverage=0.1,
        recommendation=RecommendationAction.CLARIFY,
        reasoning="Query is ambiguous",
        issues=["Poor keyword match: 0.10"],
        missing_aspects=["apply", "application"],
        relevance_scores=[0.1],
        metrics={},
    )

    message = helper.generate_clarification(
        "How do I apply?",
        mock_eval,
    )
    print(message)

    # Test 3: Max attempts reached
    print("\n\nTest 3: Max Refinement Attempts")
    print("-" * 70)

    message = helper._max_attempts_message(
        "Tell me about xyz",
        context_hint="company documents"
    )
    print(message)

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
```

**Action**: Create and test:
```bash
cd backend
python -m src.services.clarification_helper
```

---

## 🏗️ Part 2: Integrate Clarification into agent_tools.py

### Step 2.1: Import clarification helper

**File**: `backend/src/services/agent_tools.py`

**Add to imports**:
```python
from src.services.clarification_helper import ClarificationHelper
```

### Step 2.2: Add clarification logic after refinement

**In `execute_search_documents()`, after the refinement block**, add this:

```python
        # Week 2: Handle CLARIFY recommendation
        if Config.USE_SELF_REFLECTION and eval_result.should_clarify:
            clarifier = ClarificationHelper(openai_client=context.get("openai_client"))

            refinement_attempted = context.get("_refinement_count", 0) > 0

            clarification_msg = clarifier.generate_clarification(
                query=context.get("_original_query", query),
                eval_result=eval_result,
                refinement_attempted=refinement_attempted,
                context_hint=f"{collection_name} documents",
            )

            print(f"[CLARIFICATION] {clarification_msg}")

            # Return clarification to user (agent will see this and pass it along)
            return clarification_msg
```

### Step 2.3: Add progressive fallback logic

**In the refinement section**, modify the max attempts check:

Find this line in the refinement block:
```python
        if Config.USE_SELF_REFLECTION and eval_result.should_refine:
            if should_refine(eval_result, context):
                # ... refinement code ...
```

**Add an else clause** after the refinement block:
```python
            else:
                # Max refinement attempts reached - fallback to clarification
                print(f"[QUERY_REFINER] Max attempts reached, falling back to clarification")

                # Override recommendation to CLARIFY
                eval_result.recommendation = RecommendationAction.CLARIFY

                # Generate clarification message
                clarifier = ClarificationHelper(openai_client=context.get("openai_client"))
                clarification_msg = clarifier.generate_clarification(
                    query=context.get("_original_query", query),
                    eval_result=eval_result,
                    refinement_attempted=True,
                    context_hint=f"{collection_name} documents",
                )

                print(f"[CLARIFICATION] {clarification_msg}")
                return clarification_msg
```

---

## 🏗️ Part 3: Configuration Updates

### Step 3.1: Add clarification settings

**File**: `backend/src/config/settings.py`

**Add**:
```python
# Clarification Settings (Week 2)
REFLECTION_AUTO_CLARIFY: bool = os.getenv("REFLECTION_AUTO_CLARIFY", "true").lower() == "true"
REFLECTION_CLARIFY_WITH_SUGGESTIONS: bool = os.getenv("REFLECTION_CLARIFY_WITH_SUGGESTIONS", "true").lower() == "true"
```

**File**: `backend/.env`

**Add**:
```bash
# Clarification Settings
REFLECTION_AUTO_CLARIFY=true                     # Enable clarification messages
REFLECTION_CLARIFY_WITH_SUGGESTIONS=true         # Include LLM-generated suggestions
```

---

## 🧪 Part 4: Comprehensive Testing

### Step 4.1: Test all scenarios

**Create**: `backend/tests/test_week2_integration.py`

```python
"""
Integration tests for Week 2 (Query Refinement + Clarification).

Tests:
1. Query refinement improves results
2. Max attempts triggers fallback
3. Clarification messages are helpful
4. Progressive fallback works correctly

Run with: pytest tests/test_week2_integration.py -v
"""

import pytest
from unittest.mock import Mock, patch
from src.services.query_refiner import QueryRefiner, should_refine, track_refinement
from src.services.clarification_helper import ClarificationHelper
from src.models.evaluation import (
    EvaluationResult,
    QualityLevel,
    RecommendationAction,
)


class TestQueryRefinement:
    """Test query refinement scenarios."""

    def test_refinement_improves_query(self):
        """Test that refinement expands abbreviations."""
        refiner = QueryRefiner(openai_client=None)

        eval_result = EvaluationResult(
            quality=QualityLevel.PARTIAL,
            confidence=0.3,
            coverage=0.2,
            recommendation=RecommendationAction.REFINE,
            reasoning="Poor quality",
            issues=["Low keyword overlap"],
            missing_aspects=["vacation", "policy"],
            relevance_scores=[],
            metrics={},
        )

        original = "Tell me about PTO"
        refined = refiner._simple_refinement(original, eval_result)

        assert "paid time off" in refined.lower()
        assert refined != original

    def test_max_attempts_prevents_loop(self):
        """Test that max attempts prevents infinite refinement."""
        eval_result = EvaluationResult(
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

        # Should allow refinement initially
        context = {"_refinement_count": 0, "_max_refinement_attempts": 3}
        assert should_refine(eval_result, context) == True

        # Should block after max attempts
        context = {"_refinement_count": 3, "_max_refinement_attempts": 3}
        assert should_refine(eval_result, context) == False

    def test_refinement_tracking(self):
        """Test that refinement tracking works correctly."""
        context = {}

        # First refinement
        track_refinement(context, "original", "refined v1")
        assert context["_refinement_count"] == 1
        assert context["_original_query"] == "original"
        assert len(context["_refinement_history"]) == 1

        # Second refinement
        track_refinement(context, "refined v1", "refined v2")
        assert context["_refinement_count"] == 2
        assert len(context["_refinement_history"]) == 2


class TestClarification:
    """Test clarification message generation."""

    def test_no_results_message(self):
        """Test clarification for no results."""
        helper = ClarificationHelper(openai_client=None)

        eval_result = EvaluationResult(
            quality=QualityLevel.POOR,
            confidence=0.0,
            coverage=0.0,
            recommendation=RecommendationAction.CLARIFY,
            reasoning="No contexts",
            issues=["No contexts retrieved"],
            missing_aspects=[],
            relevance_scores=[],
            metrics={},
        )

        message = helper.generate_clarification(
            "quantum physics",
            eval_result,
            context_hint="HR documents"
        )

        assert "couldn't find" in message.lower()
        assert "quantum physics" in message

    def test_max_attempts_message(self):
        """Test clarification after max refinements."""
        helper = ClarificationHelper(openai_client=None)

        eval_result = EvaluationResult(
            quality=QualityLevel.POOR,
            confidence=0.2,
            coverage=0.1,
            recommendation=RecommendationAction.CLARIFY,
            reasoning="Still poor",
            issues=[],
            missing_aspects=[],
            relevance_scores=[],
            metrics={},
        )

        message = helper.generate_clarification(
            "test query",
            eval_result,
            refinement_attempted=True,
        )

        assert "multiple attempts" in message.lower()
        assert "test query" in message


class TestProgressiveFallback:
    """Test progressive fallback logic."""

    def test_escalation_after_max_refinements(self):
        """Test that system escalates to clarification after max refinements."""
        # Simulate 3 refinement attempts
        context = {}

        for i in range(3):
            track_refinement(context, f"query v{i}", f"query v{i+1}")

        eval_result = EvaluationResult(
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

        context["_max_refinement_attempts"] = 3

        # Should not allow more refinements
        assert should_refine(eval_result, context) == False
```

**Run tests**:
```bash
cd backend
pytest tests/test_week2_integration.py -v
```

---

## 🧪 Part 5: End-to-End Manual Testing

### Test Scenario 1: Successful Refinement

**Query**: "Tell me about PTO"
**Expected**:
1. First search returns poor results
2. System refines to "paid time off vacation policy"
3. Second search returns good results
4. User gets answer

**Verification**:
```
[SELF-REFLECTION] Quality: partial, Recommendation: refine
[QUERY_REFINER] Refinement attempt 1
[QUERY_REFINER] Refined: paid time off vacation policy
[SELF-REFLECTION] (After Refinement) Quality: good, Recommendation: answer
```

### Test Scenario 2: Max Attempts → Clarification

**Query**: "xyz abc def" (nonsense query)
**Expected**:
1. Attempt 1: REFINE → still poor
2. Attempt 2: REFINE → still poor
3. Attempt 3: REFINE → still poor
4. Max attempts → CLARIFY message

**Verification**:
```
[QUERY_REFINER] Refinement attempt 1
[QUERY_REFINER] Refinement attempt 2
[QUERY_REFINER] Refinement attempt 3
[QUERY_REFINER] Max attempts reached, falling back to clarification
[CLARIFICATION] After multiple attempts, I couldn't find relevant information...
```

### Test Scenario 3: Direct Clarification

**Query**: "How do I apply?"
**Expected**:
1. First search returns no results
2. System immediately clarifies (no refinement)
3. User sees helpful clarification message

**Verification**:
```
[SELF-REFLECTION] Quality: poor, Recommendation: clarify
[CLARIFICATION] I couldn't find any relevant information for: "How do I apply?"
Suggestions:
- Job application process
- Leave request submission
- Benefits enrollment
```

---

## ✅ Day 4 Completion Checklist

- [ ] `clarification_helper.py` created and tested
- [ ] ClarificationHelper generates helpful messages
- [ ] LLM-based suggestions work (when client available)
- [ ] Fallback clarification messages work (without LLM)
- [ ] agent_tools.py handles CLARIFY recommendation
- [ ] Progressive fallback implemented (max attempts → clarify)
- [ ] Configuration added for clarification settings
- [ ] Unit tests passing (test_week2_integration.py)
- [ ] Manual tests validate all scenarios:
  - [ ] Successful refinement
  - [ ] Max attempts → clarification
  - [ ] Direct clarification (no refinement)
- [ ] Logs clearly show decision flow

---

## ✅ Week 2 Complete Checklist

- [ ] **Query Refinement**:
  - [ ] QueryRefiner service working
  - [ ] Automatic refinement on REFINE recommendation
  - [ ] Loop prevention (max 3 attempts)
  - [ ] Refinement tracking in context

- [ ] **Clarification**:
  - [ ] ClarificationHelper generates messages
  - [ ] Direct clarification on CLARIFY recommendation
  - [ ] Progressive fallback after max refinements

- [ ] **Integration**:
  - [ ] agent_tools.py handles both REFINE and CLARIFY
  - [ ] Configuration controls refinement behavior
  - [ ] Logs show clear decision flow

- [ ] **Testing**:
  - [ ] Unit tests for refiner and clarifier
  - [ ] Integration tests for Week 2 features
  - [ ] Manual tests validate user experience

---

## 🎯 Success Criteria

Week 2 is complete when:

✅ Poor queries are automatically refined and retried
✅ Ambiguous queries generate helpful clarification messages
✅ Max refinement attempts trigger fallback to clarification
✅ No infinite refinement loops occur
✅ All tests passing
✅ Logs show clear refinement/clarification flow
✅ Configuration allows enabling/disabling features

---

## 🚀 What's Next: Week 3

### External Search Integration

**Goals**:
- Integrate with external search APIs (Brave Search via MCP)
- Handle EXTERNAL recommendation
- Merge internal + external results
- Provide attribution for external sources

**Timeline**: 2-3 days

---

**Congratulations on completing Week 2! 🎉**

Your agent now intelligently refines queries and asks for clarification when needed, significantly improving retrieval quality and user experience.
