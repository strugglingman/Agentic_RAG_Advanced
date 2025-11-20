# Day 2 Implementation Checklist

Quick step-by-step checklist for implementing the RetrievalEvaluator.

---

## Prerequisites ✅

- [x] Day 1 complete (evaluation.py working)
- [x] All tests passing: `python -m src.models.evaluation`
- [x] Configuration loaded from settings.py
- [x] OpenAI client available in your app

---

## Part 1: Create RetrievalEvaluator Class

### File: `src/services/retrieval_evaluator.py`

- [ ] Create file `retrieval_evaluator.py`
- [ ] Add module docstring
- [ ] Import required modules:
  - [ ] `from src.models.evaluation import *`
  - [ ] `from openai import OpenAI`
  - [ ] `from typing import List, Dict, Any, Optional, Tuple`
  - [ ] `import re, json`
- [ ] Define `RetrievalEvaluator` class
- [ ] Implement `__init__(self, config, openai_client)`
  - [ ] Store config
  - [ ] Store openai_client
  - [ ] Validate: if mode needs LLM, client must not be None
- [ ] Implement `evaluate(self, criteria)` method
  - [ ] Route to _evaluate_fast() / _evaluate_balanced() / _evaluate_thorough()
  - [ ] Based on criteria.mode or config.mode

---

## Part 2: Implement FAST Mode (Heuristics Only)

### Method: `_evaluate_fast(self, criteria: EvaluationCriteria) -> EvaluationResult`

- [ ] Extract basic metrics:
  - [ ] `context_count = len(criteria.contexts)`
  - [ ] `query_keywords` from criteria.query
  - [ ] `all_context_text` from criteria.contexts

- [ ] Calculate keyword overlap:
  - [ ] Define stopwords list
  - [ ] Filter query keywords (remove stopwords)
  - [ ] Count keywords found in contexts
  - [ ] Calculate `keyword_overlap = found / total`

- [ ] Calculate retrieval scores:
  - [ ] Extract scores from contexts (if available)
  - [ ] Calculate `avg_score`
  - [ ] Calculate `min_score`

- [ ] Calculate confidence score:
  - [ ] Formula: `keyword_overlap * 0.4 + avg_score * 0.3 + min_score * 0.2 + context_presence * 0.1`
  - [ ] Clamp to [0.0, 1.0]

- [ ] Calculate coverage:
  - [ ] Formula: `keyword_overlap * min(1.0, avg_score * 1.2)`

- [ ] Detect issues:
  - [ ] No contexts retrieved?
  - [ ] Too few contexts?
  - [ ] Low average score?
  - [ ] Poor keyword overlap?

- [ ] Identify missing aspects:
  - [ ] Keywords not found in any context

- [ ] Determine recommendation:
  - [ ] ANSWER: confidence >= excellent threshold
  - [ ] REFINE: partial confidence
  - [ ] EXTERNAL: no contexts or very poor
  - [ ] CLARIFY: low confidence with contexts

- [ ] Build EvaluationResult:
  - [ ] quality = config.get_quality_level(confidence)
  - [ ] Set all fields
  - [ ] mode_used = ReflectionMode.FAST

- [ ] Return result

---

## Part 3: Implement BALANCED Mode (Heuristics + LLM)

### Method: `_evaluate_balanced(self, criteria) -> EvaluationResult`

- [ ] Call `_evaluate_fast()` first to get baseline
- [ ] Check if confidence is borderline (0.5 < conf < 0.75)
- [ ] If borderline:
  - [ ] Call `_quick_llm_check()` to validate
  - [ ] Adjust confidence based on LLM response
  - [ ] Update reasoning
- [ ] Update `mode_used = ReflectionMode.BALANCED`
- [ ] Return updated result

### Helper: `_quick_llm_check(self, query, contexts, baseline_conf) -> (float, str)`

- [ ] Format contexts into readable text
- [ ] Build prompt:
  - [ ] Include query
  - [ ] Include contexts
  - [ ] Ask: "Can these answer the query? (yes/partial/no)"
- [ ] Call OpenAI API:
  - [ ] Model: gpt-4o-mini
  - [ ] Temperature: 0.0
  - [ ] Max tokens: 100-200
- [ ] Parse response:
  - [ ] Extract answer (yes/partial/no)
  - [ ] Extract reasoning
  - [ ] Adjust confidence: yes → +0.1, no → -0.1, partial → no change
- [ ] Return (adjusted_confidence, llm_reasoning)

---

## Part 4: Implement THOROUGH Mode (Full LLM)

### Method: `_evaluate_thorough(self, criteria) -> EvaluationResult`

- [ ] Format query + contexts for LLM
- [ ] Build detailed evaluation prompt:
  - [ ] Ask for relevance scores per context
  - [ ] Ask for coverage score
  - [ ] Ask for confidence score
  - [ ] Ask for issues list
  - [ ] Ask for missing aspects
  - [ ] Ask for recommendation
  - [ ] Request JSON format

- [ ] Call OpenAI API:
  - [ ] Model: gpt-4o-mini or gpt-4o
  - [ ] Temperature: 0.0
  - [ ] Response format: JSON (use json_object mode if available)

- [ ] Parse LLM JSON response:
  - [ ] Extract all fields
  - [ ] Convert recommendation string to RecommendationAction enum
  - [ ] Validate scores are in [0.0, 1.0]

- [ ] Build EvaluationResult:
  - [ ] Use LLM-provided values
  - [ ] quality = config.get_quality_level(confidence)
  - [ ] mode_used = ReflectionMode.THOROUGH

- [ ] Error handling:
  - [ ] If LLM fails, fallback to _evaluate_balanced()
  - [ ] If JSON parsing fails, fallback to _evaluate_balanced()

- [ ] Return result

---

## Part 5: Integration with Agent Tools

### File: `src/services/agent_tools.py`

#### Modify `execute_search_documents()`:

- [ ] Import evaluator classes at top:
  ```python
  from src.services.retrieval_evaluator import RetrievalEvaluator
  from src.models.evaluation import ReflectionConfig, EvaluationCriteria
  ```

- [ ] After `results = retrieve(...)`:
  - [ ] Check if `Config.USE_SELF_REFLECTION` is enabled
  - [ ] If enabled:
    - [ ] Load config: `config = ReflectionConfig.from_settings(Config)`
    - [ ] Get OpenAI client from context
    - [ ] Create evaluator
    - [ ] Build EvaluationCriteria
    - [ ] Call evaluator.evaluate()
    - [ ] Store result in context: `context["_last_evaluation"] = eval_result`
    - [ ] Log evaluation result

- [ ] Continue with existing formatting logic
- [ ] Return formatted results (unchanged)

---

## Part 6: Pass OpenAI Client Through Context

### Find where Agent is created (likely in routes/chat.py):

- [ ] Locate route handler that creates Agent
- [ ] Find where context dict is built
- [ ] Add OpenAI client to context:
  ```python
  context = {
      "collection": collection,
      "dept_id": dept_id,
      "user_id": user_id,
      "openai_client": client,  # ADD THIS LINE
      "request_data": request_data,
  }
  ```

---

## Part 7: Testing

### Unit Tests (Optional but Recommended):

- [ ] Create `tests/test_retrieval_evaluator.py`
- [ ] Test keyword overlap calculation
- [ ] Test confidence scoring
- [ ] Test FAST mode end-to-end
- [ ] Test BALANCED mode with mocked LLM
- [ ] Test THOROUGH mode with mocked LLM

### Manual Testing:

- [ ] Enable self-reflection in .env:
  ```bash
  USE_SELF_REFLECTION=true
  REFLECTION_MODE=fast
  ```

- [ ] Restart backend server

- [ ] Test good query (should return ANSWER):
  ```bash
  curl -X POST http://localhost:5000/chat/agent \
    -H "Content-Type: application/json" \
    -d '{"message": "What is our vacation policy?", "collection_name": "hr_docs"}'
  ```
  - [ ] Check logs for evaluation output
  - [ ] Verify: quality=good/excellent, recommendation=answer

- [ ] Test bad query (should return EXTERNAL or CLARIFY):
  ```bash
  curl -X POST http://localhost:5000/chat/agent \
    -H "Content-Type: application/json" \
    -d '{"message": "What is quantum physics?", "collection_name": "hr_docs"}'
  ```
  - [ ] Check logs for evaluation output
  - [ ] Verify: quality=poor, recommendation=external or clarify

- [ ] Test BALANCED mode:
  - [ ] Change .env: `REFLECTION_MODE=balanced`
  - [ ] Restart server
  - [ ] Repeat tests above
  - [ ] Verify LLM check runs for borderline cases

- [ ] Test THOROUGH mode:
  - [ ] Change .env: `REFLECTION_MODE=thorough`
  - [ ] Restart server
  - [ ] Repeat tests above
  - [ ] Verify detailed LLM evaluation runs

---

## Part 8: Verification & Documentation

- [ ] All three modes working correctly
- [ ] Evaluation results visible in logs
- [ ] Agent behavior unchanged (Week 1: evaluation only)
- [ ] No errors in console
- [ ] OpenAI API calls successful (for BALANCED/THOROUGH)

### Code Review:

- [ ] Code follows project style (PEP 8, type hints)
- [ ] Docstrings present for all methods
- [ ] Error handling in place
- [ ] No hardcoded values (use Config)
- [ ] Logging statements helpful for debugging

### Documentation:

- [ ] Add usage example to README (optional)
- [ ] Document any configuration changes
- [ ] Update .env.example if needed

---

## Success Criteria ✅

Before considering Day 2 complete:

- [ ] `retrieval_evaluator.py` exists and is functional
- [ ] All three modes (FAST/BALANCED/THOROUGH) implemented
- [ ] Integration with agent_tools.py complete
- [ ] Manual testing passes for good/bad queries
- [ ] Evaluation results logged correctly
- [ ] No errors or warnings in console
- [ ] OpenAI client properly passed through context
- [ ] Week 1 behavior maintained: evaluation runs but doesn't change agent actions

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

**Ready to start? Begin with Part 1! 🚀**
