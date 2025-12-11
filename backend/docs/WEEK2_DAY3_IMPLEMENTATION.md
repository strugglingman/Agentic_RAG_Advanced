# Week 2 - Day 3: Query Refinement Implementation

**Status**: ✅ COMPLETE
**Focus**: Core query refinement logic with loop prevention

---

## 📋 What Was Built

### 1. QueryRefiner Service (`src/services/query_refiner.py`)

**Class**: `QueryRefiner`

```python
class QueryRefiner:
    def __init__(self, openai_client, model="gpt-4o-mini", temperature=0.3):
        self.client = openai_client
        self.model = model
        self.temperature = temperature

    def refine_query(self, original_query, eval_result, context_hint=None) -> str:
        """
        Refine query using LLM, with fallback to simple refinement.

        Flow:
        1. If no client → use _simple_refinement()
        2. Build prompt with issues + missing_aspects
        3. Call LLM
        4. Validate result (not too short, not same as original)
        5. On error → fallback to _simple_refinement()
        """

    def _build_refinement_prompt(self, original_query, eval_result, context_hint) -> str:
        """Build LLM prompt with evaluation feedback."""

    def _simple_refinement(self, original_query, eval_result) -> str:
        """Fallback: append missing keywords to query."""
```

**Helper Functions**:
```python
def should_refine(eval_result, context) -> bool:
    """Check if recommendation is REFINE and under max attempts."""

def track_refinement(context, original_query, refined_query):
    """Track refinement history in context dict."""
```

---

### 2. Integration in agent_tools.py

**Location**: `execute_search_documents()` (lines 156-242)

**Flow**:
```
1. Initial retrieval with original query
2. Evaluate quality with RetrievalEvaluator
3. If USE_SELF_REFLECTION and REFLECTION_AUTO_REFINE enabled:
   └─ While eval_result.should_refine AND refinement_count < max_attempts:
       a. Increment refinement_count
       b. Refine query using QueryRefiner
       c. Track history
       d. Retrieve with refined query
       e. Re-evaluate quality
       f. Log results
4. Return best contexts found
```

**Key Implementation Details**:
- Uses **local counter** (`refinement_count`) - NOT context-based tracking
- Uses `eval_result.should_refine` property (not helper function)
- Passes `context_hint` with first 100 chars of each context chunk
- Handles empty results by keeping previous contexts
- Logs each attempt with `[QUERY_REFINER]` and `[SELF-REFLECTION]` prefixes

---

### 3. Configuration

**settings.py**:
```python
REFLECTION_AUTO_REFINE = os.getenv("REFLECTION_AUTO_REFINE", "true").lower() in {"1", "true", "yes"}
REFLECTION_MAX_REFINEMENT_ATTEMPTS = int(os.getenv("REFLECTION_MAX_REFINEMENT_ATTEMPTS", "3"))
```

**.env**:
```bash
REFLECTION_AUTO_REFINE=true
REFLECTION_MAX_REFINEMENT_ATTEMPTS=3
```

---

## 📁 Files Modified/Created

| File | Status | Description |
|------|--------|-------------|
| `src/services/query_refiner.py` | ✅ Created | QueryRefiner class + helpers |
| `src/services/agent_tools.py` | ✅ Modified | Added refinement loop |
| `src/config/settings.py` | ✅ Modified | Added refinement config |
| `.env` | ✅ Modified | Added refinement settings |

---

## 🔍 Actual Code Structure

### query_refiner.py Structure:
```
Lines 1-14:    Imports
Lines 16-50:   QueryRefiner.__init__()
Lines 52-117:  QueryRefiner.refine_query()
Lines 119-188: QueryRefiner._build_refinement_prompt()
Lines 190-236: QueryRefiner._simple_refinement()
Lines 244-274: should_refine() helper
Lines 277-317: track_refinement() helper
Lines 324-470: Test block
```

### agent_tools.py Integration:
```
Line 22:       Import QueryRefiner
Lines 156-165: Initialize refiner if AUTO_REFINE enabled
Lines 167-168: Initialize refinement_history list
Lines 170-232: While loop for refinement
Lines 234-242: Final logging
```

---

## 🧪 Testing

### Run QueryRefiner Tests:
```bash
cd backend
python -m src.services.query_refiner
```

**Expected Output**:
```
======================================================================
QUERY REFINER TEST
======================================================================
[OK] QueryRefiner created

----------------------------------------------------------------------
Test 1: Query Refinement with LLM
----------------------------------------------------------------------
  Original: Tell me about PTO
  Refined:  employee paid time off vacation policy benefits
  [OK] LLM refinement complete

----------------------------------------------------------------------
Test 2: Simple Refinement (fallback)
----------------------------------------------------------------------
  Original: Tell me about PTO
  Simple:   tell me about pto vacation time off policy
  [OK] Simple refinement complete

----------------------------------------------------------------------
Test 3: should_refine Logic
----------------------------------------------------------------------
  Config.REFLECTION_MAX_REFINEMENT_ATTEMPTS = 3
  Count=0: should_refine=True (expected: True)
  Count=3: should_refine=False (expected: False)
  Recommendation=ANSWER: should_refine=False (expected: False)
  [OK] should_refine logic verified

----------------------------------------------------------------------
Test 4: track_refinement
----------------------------------------------------------------------
[QUERY_REFINER] Refinement attempt 1
[QUERY_REFINER] Original: original query
[QUERY_REFINER] Refined: refined query v1
[QUERY_REFINER] Refinement attempt 2
[QUERY_REFINER] Original: refined query v1
[QUERY_REFINER] Refined: refined query v2
  Refinement count: 2 (expected: 2)
  History length: 2 (expected: 2)
  [OK] track_refinement verified

======================================================================
ALL TESTS COMPLETE!
======================================================================
```

---

## 📊 Log Output Example

When refinement triggers during actual usage:

```
[SELF-REFLECTION] Quality: partial, Confidence: 0.35, Recommendation: refine
[SELF-REFLECTION] Reasoning: Partial confidence - query refinement may help
[SELF-REFLECTION] Issues: Low average relevance score: 0.32, Poor keyword match: 0.20
[SELF-REFLECTION] Missing aspects: vacation, policy, time

[QUERY_REFINER] Refinement attempt 1/3
[QUERY_REFINER] 'Tell me about PTO' -> 'employee paid time off vacation policy'

[SELF-REFLECTION] (Attempt 1) Quality: good, Confidence: 0.78, Recommendation: answer
[QUERY_REFINER] Refinement complete after 1 attempt(s)
```

---

## ✅ Day 3 Checklist - ALL COMPLETE

- [x] `query_refiner.py` created
- [x] `QueryRefiner` class with LLM refinement
- [x] `_simple_refinement()` fallback method
- [x] `_build_refinement_prompt()` with issues/missing_aspects
- [x] `should_refine()` helper function
- [x] `track_refinement()` helper function
- [x] Integration in `agent_tools.py`
- [x] Refinement loop with max attempts
- [x] Configuration in settings.py
- [x] Environment variables in .env
- [x] Test block with 4 test cases
- [x] Logging for debugging

---

## ⚠️ Note: Difference from Original Doc

The actual implementation differs slightly from the original planning doc:

| Original Doc | Actual Implementation |
|--------------|----------------------|
| Uses `should_refine()` helper in agent_tools | Uses `eval_result.should_refine` property directly |
| Uses `track_refinement()` helper | Uses local `refinement_history` list |
| Context-based `_refinement_count` | Local `refinement_count` variable |

**Why the difference**: The actual implementation is simpler and self-contained within the tool execution, avoiding potential state issues across tool calls.

---

## 🚀 Next: Day 4

Day 4 will add:
- Clarification helper (for CLARIFY recommendation)
- Progressive fallback (max refinements → clarification message)
- Comprehensive testing
