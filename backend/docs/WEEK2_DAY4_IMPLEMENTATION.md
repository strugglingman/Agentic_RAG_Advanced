# Week 2 - Day 4: Clarification & Progressive Fallback

**Status**: 📋 TODO
**Focus**: Clarification logic, progressive fallback, comprehensive testing
**Time**: 3-4 hours
**Prerequisites**: Day 3 Complete ✅

---

## 📋 Overview

Day 4 implements the remaining Week 2 features:
1. **Clarification logic**: Generate helpful messages when queries are ambiguous
2. **Progressive fallback**: After max refinement attempts → clarification message
3. **Comprehensive testing**: Validate all refinement + clarification scenarios

---

## 🎯 What to Build

### Current State (After Day 3):

```python
# agent_tools.py - Current behavior
while eval_result.should_refine and refinement_count < max_attempts:
    # Refine and retry
    ...

# After max attempts: just returns whatever contexts were found
# NO clarification message if still poor quality
```

### Target State (After Day 4):

```python
# agent_tools.py - With clarification fallback
while eval_result.should_refine and refinement_count < max_attempts:
    # Refine and retry
    ...

# NEW: After max attempts, if still poor → return clarification message
if refinement_count >= max_attempts and eval_result.should_refine:
    return clarification_helper.generate_clarification(query, eval_result)
```

---

## 🏗️ Part 1: Create ClarificationHelper Service

### Step 1.1: Create clarification_helper.py

**File**: `backend/src/services/clarification_helper.py`

```python
"""
Clarification message generation for self-reflection system.

Generates helpful messages when queries fail to retrieve relevant
information after refinement attempts.
"""

from typing import Optional, List
from openai import OpenAI
from src.models.evaluation import EvaluationResult


class ClarificationHelper:
    """
    Generates clarification messages for ambiguous/failed queries.

    When retrieval fails or max refinements reached, this class
    generates helpful messages asking the user for more context.
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
        max_attempts_reached: bool = False,
        context_hint: Optional[str] = None,
    ) -> str:
        """
        Generate a clarification message.

        Args:
            query: Original user query
            eval_result: Evaluation result with issues
            refinement_attempted: Whether refinement was tried
            max_attempts_reached: Whether max refinements reached
            context_hint: Optional hint about document collection

        Returns:
            Clarification message string
        """
        if max_attempts_reached:
            return self._max_attempts_message(query, eval_result, context_hint)
        elif not eval_result.issues or "No contexts" in str(eval_result.issues):
            return self._no_results_message(query, context_hint)
        else:
            return self._ambiguous_query_message(query, eval_result, context_hint)

    def _no_results_message(
        self,
        query: str,
        context_hint: Optional[str] = None,
    ) -> str:
        """Generate message when no contexts found."""
        message = (
            f"I couldn't find any relevant information for: \"{query}\"\n\n"
            "This could mean:\n"
            "- The information isn't in the uploaded documents\n"
            "- Try using different keywords\n"
            "- Be more specific about what you're looking for"
        )

        if context_hint:
            message += f"\n\nNote: I searched in {context_hint}"

        return message

    def _ambiguous_query_message(
        self,
        query: str,
        eval_result: EvaluationResult,
        context_hint: Optional[str] = None,
    ) -> str:
        """Generate message for ambiguous queries."""
        message = f"Your query \"{query}\" returned low-quality results.\n\n"

        if eval_result.issues:
            message += "Issues found:\n"
            for issue in eval_result.issues[:3]:  # Max 3 issues
                message += f"- {issue}\n"

        if eval_result.missing_aspects:
            message += f"\nMissing keywords: {', '.join(eval_result.missing_aspects[:5])}\n"

        message += "\nPlease try:\n"
        message += "- Being more specific\n"
        message += "- Using different keywords\n"
        message += "- Providing more context"

        return message

    def _max_attempts_message(
        self,
        query: str,
        eval_result: EvaluationResult,
        context_hint: Optional[str] = None,
    ) -> str:
        """Generate message after max refinement attempts."""
        message = (
            f"After multiple search attempts, I couldn't find highly relevant "
            f"information for: \"{query}\"\n\n"
            "The best results I found may not fully answer your question.\n\n"
            "Suggestions:\n"
            "- Try rephrasing your question\n"
            "- Break it into smaller, more specific questions\n"
            "- Check if the information exists in the uploaded documents"
        )

        if eval_result.missing_aspects:
            message += f"\n\nKeywords not found: {', '.join(eval_result.missing_aspects[:5])}"

        if context_hint:
            message += f"\n\nSearched in: {context_hint}"

        return message


# =============================================================================
# TESTING (run with: python -m src.services.clarification_helper)
# =============================================================================

if __name__ == "__main__":
    from src.models.evaluation import (
        EvaluationResult,
        QualityLevel,
        RecommendationAction,
    )

    print("=" * 70)
    print("CLARIFICATION HELPER TEST")
    print("=" * 70)

    helper = ClarificationHelper(openai_client=None)

    # Test 1: No results
    print("\n" + "-" * 70)
    print("Test 1: No Results Message")
    print("-" * 70)
    message = helper._no_results_message(
        "What is quantum physics?",
        context_hint="HR documents"
    )
    print(message)

    # Test 2: Ambiguous query
    print("\n" + "-" * 70)
    print("Test 2: Ambiguous Query Message")
    print("-" * 70)
    mock_eval = EvaluationResult(
        quality=QualityLevel.POOR,
        confidence=0.2,
        coverage=0.1,
        recommendation=RecommendationAction.CLARIFY,
        reasoning="Poor quality",
        issues=["Low average relevance score: 0.20", "Poor keyword match: 0.10"],
        missing_aspects=["apply", "application", "process", "submit"],
        relevance_scores=[0.1, 0.2],
        metrics={},
    )
    message = helper._ambiguous_query_message(
        "How do I apply?",
        mock_eval,
    )
    print(message)

    # Test 3: Max attempts
    print("\n" + "-" * 70)
    print("Test 3: Max Attempts Message")
    print("-" * 70)
    message = helper._max_attempts_message(
        "Tell me about xyz",
        mock_eval,
        context_hint="company documents"
    )
    print(message)

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE!")
    print("=" * 70)
```

---

## 🏗️ Part 2: Integrate Clarification into agent_tools.py

### Step 2.1: Add import

**File**: `backend/src/services/agent_tools.py`

**Add to imports** (around line 22):
```python
from src.services.clarification_helper import ClarificationHelper
```

### Step 2.2: Add clarification after refinement loop

**Location**: After the refinement loop (around line 243), before formatting contexts

**Find this code** (around line 234-242):
```python
                # Log final status
                if refinement_count > 0:
                    if refinement_count >= max_attempts and eval_result.should_refine:
                        print(f"[QUERY_REFINER] Max refinement attempts ({max_attempts}) reached")
                    else:
                        print(f"[QUERY_REFINER] Refinement complete after {refinement_count} attempt(s)")

                    # Store history in context for debugging (optional)
                    context["_refinement_history"] = refinement_history
```

**Add clarification logic after** (insert around line 243):
```python
                # Progressive fallback: if still poor after max attempts, return clarification
                if refinement_count >= max_attempts and eval_result.should_clarify:
                    clarifier = ClarificationHelper(openai_client=client)
                    clarification_msg = clarifier.generate_clarification(
                        query=query,  # Original query
                        eval_result=eval_result,
                        max_attempts_reached=True,
                        context_hint=f"documents in collection",
                    )
                    print(f"[CLARIFICATION] Returning clarification message")
                    # Still return contexts, but prepend clarification
                    # This lets the agent know the quality is poor
```

**Note**: The exact integration depends on your preference:
- **Option A**: Return clarification message instead of contexts (strict)
- **Option B**: Prepend clarification to contexts (let agent decide)

---

## 🏗️ Part 3: Configuration (Optional)

**If you want to control clarification behavior**:

**settings.py**:
```python
REFLECTION_AUTO_CLARIFY = os.getenv("REFLECTION_AUTO_CLARIFY", "true").lower() in {"1", "true", "yes"}
```

**.env**:
```bash
REFLECTION_AUTO_CLARIFY=true
```

---

## 🧪 Part 4: Testing

### Test 1: Run ClarificationHelper standalone

```bash
cd backend
python -m src.services.clarification_helper
```

### Test 2: Integration test scenarios

**Scenario A: Successful refinement (no clarification)**
```
Query: "Tell me about PTO"
→ Initial eval: REFINE
→ Refinement: "paid time off vacation policy"
→ New eval: ANSWER
→ Result: Returns contexts (no clarification needed)
```

**Scenario B: Max attempts reached → clarification**
```
Query: "xyz abc nonsense"
→ Initial eval: REFINE
→ Refinement 1: still poor
→ Refinement 2: still poor
→ Refinement 3: still poor
→ Max attempts reached
→ Result: Returns clarification message
```

**Scenario C: Direct clarification (no contexts)**
```
Query: "What is the weather?"
→ No contexts found
→ Eval: CLARIFY (not REFINE)
→ Result: Returns clarification message immediately
```

---

## 📁 Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/services/clarification_helper.py` | CREATE | ClarificationHelper class |
| `src/services/agent_tools.py` | MODIFY | Add clarification import + logic |
| `src/config/settings.py` | MODIFY (optional) | Add REFLECTION_AUTO_CLARIFY |
| `.env` | MODIFY (optional) | Add clarification setting |

---

## ✅ Day 4 Checklist

- [ ] Create `clarification_helper.py`
- [ ] Implement `ClarificationHelper` class
- [ ] Implement `_no_results_message()`
- [ ] Implement `_ambiguous_query_message()`
- [ ] Implement `_max_attempts_message()`
- [ ] Add test block
- [ ] Run standalone tests
- [ ] Import in `agent_tools.py`
- [ ] Add clarification logic after refinement loop
- [ ] Test Scenario A: refinement success
- [ ] Test Scenario B: max attempts → clarification
- [ ] Test Scenario C: direct clarification
- [ ] (Optional) Add configuration settings

---

## 🎯 Success Criteria

Day 4 is complete when:

- [ ] ClarificationHelper generates helpful messages
- [ ] Max refinement attempts triggers clarification
- [ ] Direct CLARIFY recommendation returns message
- [ ] All test scenarios pass
- [ ] Logs show clear decision flow

---

## 📊 Expected Log Output

### Scenario B: Max attempts → clarification
```
[SELF-REFLECTION] Quality: partial, Confidence: 0.35, Recommendation: refine

[QUERY_REFINER] Refinement attempt 1/3
[QUERY_REFINER] 'xyz query' -> 'xyz expanded query'
[SELF-REFLECTION] (Attempt 1) Quality: partial, Confidence: 0.38, Recommendation: refine

[QUERY_REFINER] Refinement attempt 2/3
[QUERY_REFINER] 'xyz expanded query' -> 'xyz more keywords'
[SELF-REFLECTION] (Attempt 2) Quality: partial, Confidence: 0.40, Recommendation: refine

[QUERY_REFINER] Refinement attempt 3/3
[QUERY_REFINER] 'xyz more keywords' -> 'xyz final attempt'
[SELF-REFLECTION] (Attempt 3) Quality: partial, Confidence: 0.42, Recommendation: refine

[QUERY_REFINER] Max refinement attempts (3) reached
[CLARIFICATION] Returning clarification message
```

---

## 🚀 After Day 4: Week 2 Complete!

Once Day 4 is done, Week 2 features are complete:

| Feature | Day | Status |
|---------|-----|--------|
| QueryRefiner service | Day 3 | ✅ Done |
| Refinement loop in agent_tools | Day 3 | ✅ Done |
| ClarificationHelper service | Day 4 | 📋 TODO |
| Progressive fallback | Day 4 | 📋 TODO |
| Comprehensive testing | Day 4 | 📋 TODO |

---

## 📚 References

- Day 3 Implementation: [WEEK2_DAY3_IMPLEMENTATION.md](./WEEK2_DAY3_IMPLEMENTATION.md)
- Query Refiner: [../src/services/query_refiner.py](../src/services/query_refiner.py)
- Agent Tools: [../src/services/agent_tools.py](../src/services/agent_tools.py)
- Evaluation Models: [../src/models/evaluation.py](../src/models/evaluation.py)
