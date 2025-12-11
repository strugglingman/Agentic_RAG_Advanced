# Week 2 Implementation Checklist

**Status**: Day 3 ✅ Complete | Day 4 📋 TODO
**Estimated Time**: 2-3 days (7-9 hours total)
**Prerequisites**: Week 1 Complete ✅

---

## Quick Reference

| Day | Focus | Time | Status |
|-----|-------|------|--------|
| **Day 3** | Query Refinement | 4-5h | ✅ **COMPLETE** |
| **Day 4** | Clarification | 3-4h | 📋 TODO |

---

## Day 3: Query Refinement ✅ COMPLETE

### Part 1: QueryRefiner Service ✅

- [x] Create `src/services/query_refiner.py`
- [x] Implement `QueryRefiner` class:
  - [x] `__init__(openai_client, model, temperature)`
  - [x] `refine_query(original_query, eval_result, context_hint)`
  - [x] `_build_refinement_prompt()`
  - [x] `_simple_refinement()` (fallback without LLM)
- [x] Implement helper functions:
  - [x] `should_refine(eval_result, context)`
  - [x] `track_refinement(context, original, refined)`
- [x] Test QueryRefiner standalone

### Part 2: Configuration ✅

- [x] Add `REFLECTION_AUTO_REFINE` to settings.py
- [x] Add `REFLECTION_MAX_REFINEMENT_ATTEMPTS` to settings.py
- [x] Add settings to `.env`

### Part 3: Integration in agent_tools.py ✅

- [x] Import QueryRefiner
- [x] Initialize refiner if `Config.REFLECTION_AUTO_REFINE` enabled
- [x] Implement refinement loop with local counter
- [x] Track refinement history
- [x] Re-evaluate after each refinement
- [x] Handle empty results (keep previous)
- [x] Log all attempts

### Part 4: Testing ✅

- [x] Test block in query_refiner.py
- [x] Test LLM refinement
- [x] Test simple refinement fallback
- [x] Test should_refine() logic
- [x] Test track_refinement()

---

## Day 4: Clarification & Testing 📋 TODO

### Part 1: ClarificationHelper Service

- [x] Create `src/services/clarification_helper.py` (skeleton with TODOs)
- [ ] Implement `ClarificationHelper` class:
  - [ ] `__init__(openai_client, model, temperature)`
  - [ ] `generate_clarification(query, eval_result, ...)`
  - [ ] `_no_results_message()`
  - [ ] `_ambiguous_query_message()`
  - [ ] `_max_attempts_message()`
- [ ] Add test block
- [ ] Run standalone tests: `python -m src.services.clarification_helper`

### Part 2: Integration in agent_tools.py

- [x] TODO block added with implementation steps (lines 244-273)
- [ ] Import ClarificationHelper
- [ ] Add clarification logic after refinement loop:
  - [ ] Check if max attempts reached AND still poor quality
  - [ ] Generate clarification message
  - [ ] Return message (or prepend to contexts)

### Part 3: Configuration (Optional)

- [ ] Add `REFLECTION_AUTO_CLARIFY` to settings.py (optional)
- [ ] Add to `.env` (optional)

### Part 4: Comprehensive Testing

- [ ] Test Scenario A: Refinement success (no clarification)
- [ ] Test Scenario B: Max attempts → clarification
- [ ] Test Scenario C: Direct clarification (no contexts)

---

## Files Summary

### Created in Day 3 ✅:
```
src/services/query_refiner.py          ✅ Created (470 lines)
```

### Modified in Day 3 ✅:
```
src/services/agent_tools.py            ✅ Modified (refinement loop)
src/config/settings.py                 ✅ Modified (refinement config)
.env                                   ✅ Modified (refinement settings)
```

### Created Skeletons for Day 4:
```
src/services/clarification_helper.py   ✅ Skeleton created (with TODOs)
src/services/agent_tools.py            ✅ TODO block added (lines 244-273)
```

### To Complete in Day 4:
```
src/services/clarification_helper.py   📋 Implement methods
src/services/agent_tools.py            📋 Implement clarification logic
```

---

## Current Implementation Details

### query_refiner.py Structure:
```
Lines 1-14:    Imports
Lines 16-50:   QueryRefiner.__init__()
Lines 52-117:  QueryRefiner.refine_query()  - LLM + fallback
Lines 119-188: QueryRefiner._build_refinement_prompt()
Lines 190-236: QueryRefiner._simple_refinement()
Lines 244-274: should_refine() helper
Lines 277-317: track_refinement() helper
Lines 324-470: Test block
```

### agent_tools.py Integration (lines 156-242):
```
Line 22:       Import QueryRefiner
Lines 156-165: Check AUTO_REFINE, initialize QueryRefiner
Lines 167-168: Initialize refinement_history
Lines 170-232: While loop:
               - Increment counter
               - Refine query
               - Track history
               - Retrieve with refined query
               - Re-evaluate
               - Log results
Lines 234-242: Final logging + store history
```

---

## Testing Commands

### Run QueryRefiner tests:
```bash
cd backend
python -m src.services.query_refiner
```

### Run ClarificationHelper tests (after Day 4):
```bash
cd backend
python -m src.services.clarification_helper
```

### Run integration test:
```bash
cd backend
python tests/test_retrieval_evaluator_integration.py
```

---

## Success Criteria

### Day 3 ✅ COMPLETE:
- [x] QueryRefiner service working
- [x] Automatic refinement on REFINE recommendation
- [x] Max attempts loop prevention
- [x] Refinement tracking/logging
- [x] Integration in agent_tools.py

### Day 4 (To complete):
- [ ] ClarificationHelper generates helpful messages
- [ ] Max refinement attempts triggers clarification
- [ ] All test scenarios pass
- [ ] Logs show clear decision flow

---

## Common Issues & Solutions

### Issue: Refinement doesn't trigger

**Check**:
1. `USE_SELF_REFLECTION=true` in .env
2. `REFLECTION_AUTO_REFINE=true` in .env
3. Evaluation returns `should_refine == True` (confidence < 0.7)

### Issue: Too many refinements

**Check**:
1. `REFLECTION_MAX_REFINEMENT_ATTEMPTS=3` in .env
2. Loop condition: `while eval_result.should_refine and refinement_count < max_attempts`

### Issue: LLM refinement failing

**Check**:
1. OpenAI client passed correctly in context
2. API key valid
3. Fallback to `_simple_refinement()` should work

---

## References

- **Day 3 Guide**: [WEEK2_DAY3_IMPLEMENTATION.md](./WEEK2_DAY3_IMPLEMENTATION.md) ✅
- **Day 4 Guide**: [WEEK2_DAY4_IMPLEMENTATION.md](./WEEK2_DAY4_IMPLEMENTATION.md) 📋
- **Overview**: [WEEK2_OVERVIEW.md](./WEEK2_OVERVIEW.md)
- **Week 1 Complete**: [DAY2_CHECKLIST.md](./DAY2_CHECKLIST.md) ✅

---

## Next Steps

**Day 4 Tasks**:
1. Create `clarification_helper.py` (copy from Day 4 guide)
2. Add integration in `agent_tools.py`
3. Test all scenarios
4. Complete Week 2! 🎉
