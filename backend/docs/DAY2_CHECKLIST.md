# Day 2 Implementation Checklist

## ✅ STATUS: COMPLETE

All Day 2 tasks have been successfully implemented and tested.

---

## Prerequisites ✅

- [x] Day 1 complete (evaluation.py working)
- [x] All tests passing: `python -m src.models.evaluation`
- [x] Configuration loaded from settings.py
- [x] OpenAI client available in your app

---

## Part 1: Create RetrievalEvaluator Class ✅

### File: `src/services/retrieval_evaluator.py`

- [x] Create file `retrieval_evaluator.py`
- [x] Add module docstring
- [x] Import required modules:
  - [x] `from src.models.evaluation import *`
  - [x] `from openai import OpenAI`
  - [x] `from typing import List, Dict, Any, Optional, Tuple`
  - [x] `import re, json`
- [x] Define `RetrievalEvaluator` class
- [x] Implement `__init__(self, config, openai_client)`
  - [x] Store config
  - [x] Store openai_client
  - [x] Validate: if mode needs LLM, client must not be None
- [x] Implement `evaluate(self, criteria)` method
  - [x] Route to _evaluate_fast() / _evaluate_balanced() / _evaluate_thorough()
  - [x] Based on criteria.mode or config.mode

---

## Part 2: Implement FAST Mode (Heuristics Only) ✅

### Method: `_evaluate_fast(self, criteria: EvaluationCriteria) -> EvaluationResult`

- [x] Extract basic metrics:
  - [x] `context_count = len(criteria.contexts)`
  - [x] `query_keywords` from criteria.query
  - [x] `all_context_text` from criteria.contexts

- [x] Calculate keyword overlap:
  - [x] Define stopwords list
  - [x] Filter query keywords (remove stopwords)
  - [x] Count keywords found in contexts
  - [x] Calculate `keyword_overlap = found / total`

- [x] Calculate retrieval scores:
  - [x] Extract scores from contexts (if available)
  - [x] Calculate `avg_score`
  - [x] Calculate `min_score`

- [x] Calculate confidence score:
  - [x] Formula: `keyword_overlap * 0.4 + avg_score * 0.3 + min_score * 0.2 + context_presence * 0.1`
  - [x] Clamp to [0.0, 1.0]

- [x] Calculate coverage:
  - [x] Formula: `keyword_overlap * min(1.0, avg_score * 1.2)`

- [x] Detect issues:
  - [x] No contexts retrieved?
  - [x] Too few contexts?
  - [x] Low average score?
  - [x] Poor keyword overlap?

- [x] Identify missing aspects:
  - [x] Keywords not found in any context

- [x] Determine recommendation:
  - [x] ANSWER: confidence >= excellent threshold
  - [x] REFINE: partial confidence
  - [x] EXTERNAL: no contexts or very poor
  - [x] CLARIFY: low confidence with contexts

- [x] Build EvaluationResult:
  - [x] quality = config.get_quality_level(confidence)
  - [x] Set all fields
  - [x] mode_used = ReflectionMode.FAST

- [x] Return result

---

## Part 3: Implement BALANCED Mode (Heuristics + LLM) ✅

### Method: `_evaluate_balanced(self, criteria) -> EvaluationResult`

- [x] Call `_evaluate_fast()` first to get baseline
- [x] Check if confidence is borderline (0.5 <= conf < 0.7)
- [x] If borderline:
  - [x] Call `_quick_llm_check()` to validate
  - [x] Adjust confidence based on LLM response
  - [x] Update reasoning
- [x] Update `mode_used = ReflectionMode.BALANCED`
- [x] Return updated result

### Helper: `_quick_llm_check(self, query, contexts, baseline_conf) -> (float, str)` ✅

- [x] Format contexts into readable text
- [x] Build prompt:
  - [x] Include query
  - [x] Include contexts
  - [x] Ask: "Can these answer the query? (yes/partial/no)"
- [x] Call OpenAI API:
  - [x] Model: gpt-4o-mini
  - [x] Temperature: 0.0
  - [x] Max tokens: 100-200
- [x] Parse response:
  - [x] Extract answer (yes/partial/no)
  - [x] Extract reasoning
  - [x] Adjust confidence: yes → +0.1, no → -0.1, partial → no change
- [x] Return (adjusted_confidence, llm_reasoning)

---

## Part 4: Implement THOROUGH Mode (Full LLM) ✅

### Method: `_evaluate_thorough(self, criteria) -> EvaluationResult`

- [x] Format query + contexts for LLM
- [x] Build detailed evaluation prompt:
  - [x] Ask for relevance scores per context
  - [x] Ask for coverage score
  - [x] Ask for confidence score
  - [x] Ask for issues list
  - [x] Ask for missing aspects
  - [x] Ask for recommendation
  - [x] Request JSON format

- [x] Call OpenAI API:
  - [x] Model: gpt-4o-mini
  - [x] Temperature: 0.0
  - [x] Response format: JSON (use json_object mode)

- [x] Parse LLM JSON response:
  - [x] Extract all fields
  - [x] Convert recommendation string to RecommendationAction enum
  - [x] Validate scores are in [0.0, 1.0]

- [x] Build EvaluationResult:
  - [x] Use LLM-provided values
  - [x] quality = config.get_quality_level(confidence)
  - [x] mode_used = ReflectionMode.THOROUGH

- [x] Error handling:
  - [x] If LLM fails, fallback to _evaluate_balanced()
  - [x] If JSON parsing fails, fallback to _evaluate_balanced()

- [x] Return result

---

## Part 5: Integration with Agent Tools ✅

### File: `src/services/agent_tools.py`

#### Modify `execute_search_documents()`:

- [x] Import evaluator classes at top
- [x] After `results = retrieve(...)`:
  - [x] Check if `Config.USE_SELF_REFLECTION` is enabled
  - [x] If enabled:
    - [x] Load config: `config = ReflectionConfig.from_settings(Config)`
    - [x] Get OpenAI client from context
    - [x] Create evaluator
    - [x] Build EvaluationCriteria
    - [x] Call evaluator.evaluate()
    - [x] Store result in context: `context["_last_evaluation"] = eval_result`
    - [x] Log evaluation result (Quality, Confidence, Recommendation, Reasoning, Issues, Missing aspects)

- [x] Continue with existing formatting logic
- [x] Return formatted results (unchanged)

---

## Part 6: Pass OpenAI Client Through Context ✅

### File: `src/routes/chat.py`

- [x] Locate route handler that creates Agent
- [x] Find where context dict is built
- [x] Add OpenAI client to context: `"openai_client": openai_client`

---

## Part 7: Testing ✅

### Unit Tests:

- [x] Test data models: `python -m src.models.evaluation` (9/9 tests pass)
- [x] Test evaluator: `python -m src.services.retrieval_evaluator` (3/3 tests pass)
- [x] Created integration test: `test_self_reflection.py`

### Integration Testing:

- [x] Enable self-reflection in .env (`USE_SELF_REFLECTION=true`)
- [x] Test FAST mode - working correctly
- [x] Test BALANCED mode - working correctly
- [x] Test THOROUGH mode - working correctly
- [x] Test good retrieval (high confidence → ANSWER)
- [x] Test poor retrieval (low confidence → CLARIFY/REFINE)
- [x] Test no contexts (→ REFINE/EXTERNAL)
- [x] Verify logs show correct evaluation output

---

## Part 8: Verification & Documentation ✅

- [x] All three modes working correctly
- [x] Evaluation results visible in logs
- [x] Agent behavior unchanged (Week 1: evaluation only)
- [x] No errors in console
- [x] OpenAI API calls successful (for BALANCED/THOROUGH)

### Code Review:

- [x] Code follows project style (PEP 8, type hints)
- [x] Docstrings present for all methods
- [x] Error handling in place
- [x] No hardcoded values (use Config - all thresholds centralized)
- [x] Logging statements helpful for debugging

### Documentation:

- [x] Updated implementation guide
- [x] Configuration documented in .env
- [x] Test scripts created

---

## Success Criteria - ALL COMPLETE ✅

Day 2 completion criteria:

- [x] `retrieval_evaluator.py` exists and is functional
- [x] All three modes (FAST/BALANCED/THOROUGH) implemented
- [x] Integration with agent_tools.py complete
- [x] Integration tests pass
- [x] Evaluation results logged correctly
- [x] No errors or warnings in console
- [x] OpenAI client properly passed through context
- [x] Week 1 behavior maintained: evaluation runs but doesn't change agent actions

---

## Common Issues & Solutions

### Issue: `ModuleNotFoundError: No module named 'src.services.retrieval_evaluator'`
**Solution**: Make sure file is created and named correctly, restart server

### Issue: `AttributeError: 'NoneType' object has no attribute 'chat'`
**Solution**: OpenAI client not passed in context, add it in route handler

### Issue: Evaluation always returns same result
**Solution**: Check if heuristic calculations are correct, add debug prints

### Issue: LLM calls timing out
**Solution**: Increase timeout, or use faster model (gpt-3.5-turbo)

### Issue: Confidence scores seem random
**Solution**: Verify keyword extraction and overlap calculation, test with known examples

---

## Next Steps

After Day 2 is complete:

1. **Monitor in production**: Watch evaluation metrics, tune thresholds if needed
2. **Collect data**: Log evaluations to analyze patterns
3. **Plan Day 3**: Design query refinement logic (Week 2)
4. **Plan Day 4**: Design external search integration (Week 3)

---

## Time Estimate

- **Part 1-2 (FAST mode)**: 2-3 hours
- **Part 3 (BALANCED mode)**: 1-2 hours
- **Part 4 (THOROUGH mode)**: 1-2 hours
- **Part 5-6 (Integration)**: 1 hour
- **Part 7 (Testing)**: 1-2 hours

**Total**: 6-10 hours (full day of focused work)

---

**✅ Day 2 Complete! All implementation and testing finished successfully.**

**Quick Test**:
```bash
cd backend
python test_self_reflection.py
```

**Next**: Week 2 - Implement action-taking based on evaluation results.
