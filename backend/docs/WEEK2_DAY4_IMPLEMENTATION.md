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
# Only handles REFINE recommendation (0.50 ≤ confidence < 0.70)
while eval_result.should_refine and refinement_count < max_attempts:
    # Refine and retry
    ...

# Problems:
# 1. Direct CLARIFY (confidence < 0.50) is NOT handled - skips refinement but no message
# 2. After max attempts: just returns whatever contexts were found
```

### Target State (After Day 4):

```python
# agent_tools.py - With full clarification support

# NEW: Handle direct CLARIFY (confidence < 0.50) BEFORE refinement
if eval_result.should_clarify:
    return clarification_helper.generate_clarification(query, eval_result, max_attempts_reached=False)

# Refinement loop (only for REFINE recommendation: 0.50 ≤ confidence < 0.70)
while eval_result.should_refine and refinement_count < max_attempts:
    # Refine and retry
    ...

# NEW: After max refinement attempts, if still poor → return clarification message
if refinement_count >= max_attempts and eval_result.should_refine:
    return clarification_helper.generate_clarification(query, eval_result, max_attempts_reached=True)
```

### Logic Flow Summary:

```
Evaluation Result:
├── confidence ≥ 0.70 (ANSWER)    → Return contexts directly
├── 0.50 ≤ confidence < 0.70 (REFINE) → Try refinement loop
│   ├── Refinement succeeds       → Return improved contexts
│   └── Max attempts reached      → Return clarification message
└── confidence < 0.50 (CLARIFY)   → Return clarification message directly (skip refinement)
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
        max_attempts_reached: bool = False,
        context_hint: Optional[str] = None,
    ) -> str:
        """
        Generate a clarification message.

        Args:
            query: Original user query
            eval_result: Evaluation result with issues
            max_attempts_reached: Whether max refinements reached (implies refinement was attempted)
            context_hint: Optional hint about document collection

        Returns:
            Clarification message string

        Scenarios:
            - max_attempts_reached=True: Refinement was tried but failed → _max_attempts_message
            - No contexts found: Nothing retrieved → _no_results_message
            - Direct CLARIFY (confidence < 0.5): Poor quality → _ambiguous_query_message
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

### Step 2.2: Add DIRECT clarification BEFORE refinement loop

**Location**: After initial evaluation, BEFORE refinement loop (around line 155)

**Purpose**: Handle CLARIFY recommendation (confidence < 0.50) directly without attempting refinement.

```python
            # Handle direct CLARIFY (confidence < 0.50) - skip refinement
            if eval_result.should_clarify:
                clarifier = ClarificationHelper(openai_client=client)
                clarification_msg = clarifier.generate_clarification(
                    query=query,
                    eval_result=eval_result,
                    max_attempts_reached=False,  # No refinement attempted
                    context_hint="documents in collection",
                )
                print(f"[CLARIFICATION] Direct clarify (confidence < 0.50)")
                # Option A: return clarification_msg
                # Option B: prepend to contexts
```

### Step 2.3: Add clarification AFTER refinement loop (progressive fallback)

**Location**: After the refinement loop (around line 268), before formatting contexts

**Purpose**: Handle case where refinement was tried but max attempts reached without success.

```python
            # Progressive fallback: if still poor after max attempts, return clarification
            if Config.REFLECTION_AUTO_REFINE:
                if refinement_count >= max_attempts and eval_result.should_refine:
                    clarifier = ClarificationHelper(openai_client=client)
                    clarification_msg = clarifier.generate_clarification(
                        query=query,  # Original query
                        eval_result=eval_result,
                        max_attempts_reached=True,  # Refinement was attempted
                        context_hint="documents in collection",
                    )
                    print(f"[CLARIFICATION] Max attempts reached, returning clarification")
                    # Option A: return clarification_msg
                    # Option B: prepend to contexts
```

### Integration Summary:

| Scenario | Location | `max_attempts_reached` |
|----------|----------|------------------------|
| Direct CLARIFY (confidence < 0.50) | Before refinement loop | `False` |
| Post-refinement fallback | After refinement loop | `True` |

**Note**: The exact return behavior depends on your preference:
- **Option A (strict)**: Return clarification message instead of contexts
- **Option B (soft)**: Prepend clarification to contexts (let agent decide)

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

**Scenario A: Good quality (ANSWER) - no clarification**
```
Query: "What is the vacation policy?"
→ Initial eval: ANSWER (confidence ≥ 0.70)
→ Result: Returns contexts directly (no refinement, no clarification)
```

**Scenario B: Successful refinement (REFINE → ANSWER)**
```
Query: "Tell me about PTO"
→ Initial eval: REFINE (confidence = 0.55)
→ Refinement: "paid time off vacation policy"
→ New eval: ANSWER (confidence = 0.78)
→ Result: Returns improved contexts (no clarification needed)
```

**Scenario C: Max attempts reached → clarification (REFINE → REFINE → ... → clarify)**
```
Query: "xyz abc nonsense"
→ Initial eval: REFINE (confidence = 0.52)
→ Refinement 1: still poor (confidence = 0.54)
→ Refinement 2: still poor (confidence = 0.53)
→ Refinement 3: still poor (confidence = 0.55)
→ Max attempts reached
→ Result: Returns clarification message (max_attempts_reached=True)
```

**Scenario D: Direct clarification (CLARIFY) - skip refinement**
```
Query: "What is the weather?"
→ Initial eval: CLARIFY (confidence = 0.25, < 0.50)
→ Result: Returns clarification message immediately (max_attempts_reached=False)
→ Note: Refinement loop is SKIPPED entirely
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

### Part 1: ClarificationHelper Service
- [x] Create `clarification_helper.py` (skeleton exists)
- [ ] Implement `_no_results_message()`
- [ ] Implement `_ambiguous_query_message()`
- [ ] Implement `_max_attempts_message()`
- [ ] Add test block
- [ ] Run standalone tests: `python -m src.services.clarification_helper`

### Part 2: Integration in agent_tools.py
- [ ] Import ClarificationHelper at top of file
- [ ] Add direct CLARIFY handling BEFORE refinement loop (confidence < 0.50)
- [ ] Add post-refinement fallback AFTER refinement loop (max attempts reached)

### Part 3: Testing
- [ ] Test Scenario A: ANSWER (good quality, no clarification)
- [ ] Test Scenario B: REFINE → ANSWER (successful refinement)
- [ ] Test Scenario C: REFINE → max attempts → clarification
- [ ] Test Scenario D: Direct CLARIFY (confidence < 0.50, skip refinement)

### Part 4: Optional
- [ ] Add `REFLECTION_AUTO_CLARIFY` config setting

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
