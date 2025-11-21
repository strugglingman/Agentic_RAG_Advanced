# Week 2 Implementation Checklist

**Status**: 📋 Ready to Start
**Estimated Time**: 2-3 days (7-9 hours total)
**Prerequisites**: Week 1 Complete ✅

---

## Quick Reference

| Day | Focus | Time | Key Deliverable |
|-----|-------|------|-----------------|
| **Day 3** | Query Refinement | 4-5h | QueryRefiner service + integration |
| **Day 4** | Clarification | 3-4h | ClarificationHelper + progressive fallback |

---

## Day 3: Query Refinement

### Part 1: QueryRefiner Service ⏱️ 2 hours

- [ ] Create `src/services/query_refiner.py`
- [ ] Implement `QueryRefiner` class:
  - [ ] `__init__(openai_client, model, temperature)`
  - [ ] `refine_query(original_query, eval_result, context_hint)`
  - [ ] `_build_refinement_prompt()`
  - [ ] `_simple_refinement()` (fallback without LLM)
- [ ] Implement helper functions:
  - [ ] `should_refine(eval_result, context)`
  - [ ] `track_refinement(context, original, refined)`
- [ ] Test QueryRefiner standalone:
  ```bash
  python -m src.services.query_refiner
  ```

### Part 2: Configuration ⏱️ 30 minutes

- [ ] Update `src/config/settings.py`:
  - [ ] Add `REFLECTION_AUTO_REFINE`
  - [ ] Add `REFLECTION_MAX_REFINEMENT_ATTEMPTS`
- [ ] Update `.env`:
  - [ ] Add refinement settings (default: enabled, max 3 attempts)
- [ ] Update `src/models/evaluation.py`:
  - [ ] Add `auto_refine: bool` to ReflectionConfig
  - [ ] Add `max_refinement_attempts: int` to ReflectionConfig
  - [ ] Update `from_settings()` method

### Part 3: Integration ⏱️ 1.5 hours

- [ ] Update `src/services/agent_tools.py`:
  - [ ] Import QueryRefiner and helpers
  - [ ] Initialize refinement tracking in context
  - [ ] Add refinement logic after evaluation:
    ```python
    if eval_result.should_refine and should_refine(eval_result, context):
        # Refine query
        # Track refinement
        # Retry search
        # Re-evaluate
    ```
  - [ ] Update logs to show refinement attempts

### Part 4: Testing ⏱️ 1 hour

- [ ] Create `tests/test_query_refiner.py`:
  - [ ] Test simple refinement (abbreviation expansion)
  - [ ] Test `should_refine()` logic
  - [ ] Test `track_refinement()` tracking
- [ ] Run unit tests:
  ```bash
  pytest tests/test_query_refiner.py -v
  ```
- [ ] Manual integration test:
  - [ ] Enable self-reflection + refinement in .env
  - [ ] Test with poor query (e.g., "Tell me about PTO")
  - [ ] Verify logs show refinement attempt
  - [ ] Verify refined query gets better results

---

## Day 4: Clarification & Testing

### Part 1: Clarification Helper ⏱️ 1.5 hours

- [ ] Create `src/services/clarification_helper.py`
- [ ] Implement `ClarificationHelper` class:
  - [ ] `__init__(openai_client, model, temperature)`
  - [ ] `generate_clarification(query, eval_result, refinement_attempted, context_hint)`
  - [ ] `_no_results_message()`
  - [ ] `_ambiguous_query_message()`
  - [ ] `_max_attempts_message()`
  - [ ] `_get_llm_suggestions()` (optional LLM-based)
- [ ] Test ClarificationHelper standalone:
  ```bash
  python -m src.services.clarification_helper
  ```

### Part 2: Clarification Integration ⏱️ 1 hour

- [ ] Update `src/services/agent_tools.py`:
  - [ ] Import ClarificationHelper
  - [ ] Add clarification logic:
    ```python
    if eval_result.should_clarify:
        clarification_msg = clarifier.generate_clarification(...)
        return clarification_msg
    ```
  - [ ] Add progressive fallback (max attempts → clarify)

### Part 3: Configuration ⏱️ 15 minutes

- [ ] Update `src/config/settings.py`:
  - [ ] Add `REFLECTION_AUTO_CLARIFY`
  - [ ] Add `REFLECTION_CLARIFY_WITH_SUGGESTIONS`
- [ ] Update `.env`:
  - [ ] Add clarification settings

### Part 4: Comprehensive Testing ⏱️ 1.5 hours

- [ ] Create `tests/test_week2_integration.py`:
  - [ ] Test query refinement scenarios
  - [ ] Test clarification scenarios
  - [ ] Test progressive fallback
  - [ ] Test max attempts prevention
- [ ] Run all Week 2 tests:
  ```bash
  pytest tests/test_query_refiner.py tests/test_week2_integration.py -v
  ```
- [ ] Manual end-to-end tests:
  - [ ] **Test 1**: Poor query → refinement → success
    - Query: "Tell me about PTO"
    - Expected: Refines to "paid time off", better results
  - [ ] **Test 2**: Nonsense query → max attempts → clarification
    - Query: "xyz abc def"
    - Expected: 3 refinements, then clarification message
  - [ ] **Test 3**: Ambiguous query → direct clarification
    - Query: "How do I apply?"
    - Expected: Clarification with suggestions

---

## Final Verification

### Code Quality

- [ ] All new files have docstrings
- [ ] All methods have type hints
- [ ] Error handling in place (LLM calls can fail)
- [ ] Logging statements helpful for debugging
- [ ] No hardcoded values (use Config)

### Testing

- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] Manual tests validate user experience
- [ ] Logs clearly show refinement/clarification flow

### Configuration

- [ ] Environment variables documented in .env
- [ ] Settings loaded correctly in Config
- [ ] ReflectionConfig includes Week 2 fields
- [ ] Can enable/disable refinement via config

### Integration

- [ ] agent_tools.py handles REFINE recommendation
- [ ] agent_tools.py handles CLARIFY recommendation
- [ ] Refinement tracking prevents infinite loops
- [ ] Progressive fallback works (REFINE → REFINE → CLARIFY)
- [ ] OpenAI client passed correctly for LLM calls

---

## Success Criteria

Week 2 is **COMPLETE** when:

✅ **Refinement works**:
- Poor queries automatically refined and retried
- Refinement improves search results
- Max 3 attempts enforced (no infinite loops)
- Logs show: original query → refined query → retry → result

✅ **Clarification works**:
- Ambiguous queries generate helpful messages
- Max refinement attempts trigger clarification
- Clarification includes suggestions (when LLM available)
- User gets actionable guidance

✅ **Tests pass**:
- Unit tests for QueryRefiner
- Unit tests for ClarificationHelper
- Integration tests for Week 2 scenarios
- Manual tests validate UX

✅ **Configuration works**:
- Can enable/disable refinement
- Can configure max attempts
- Settings load from .env correctly

---

## Common Issues & Solutions

### Issue: Refinement doesn't trigger

**Check**:
1. `USE_SELF_REFLECTION=true` in .env
2. `REFLECTION_AUTO_REFINE=true` in .env
3. Evaluation returns `should_refine == True`
4. Refinement count < max attempts
5. Add debug prints to verify

### Issue: Infinite refinement loop

**Check**:
1. `_refinement_count` being tracked correctly
2. `_max_refinement_attempts` set in context
3. `should_refine()` checks max attempts
4. Add logging to show attempt count

### Issue: Clarification messages not helpful

**Check**:
1. OpenAI client available for LLM suggestions
2. Evaluation result includes issues/missing_aspects
3. Test with different query types
4. Adjust prompt templates if needed

### Issue: LLM calls failing

**Check**:
1. OpenAI API key valid
2. OpenAI client passed in context
3. Error handling catches exceptions
4. Fallback logic works without LLM

---

## Time Breakdown

| Task | Estimated | Actual |
|------|-----------|--------|
| **Day 3: Query Refinement** | | |
| - Create QueryRefiner | 2h | |
| - Configuration | 0.5h | |
| - Integration | 1.5h | |
| - Testing | 1h | |
| **Day 3 Subtotal** | **5h** | |
| | | |
| **Day 4: Clarification** | | |
| - Create ClarificationHelper | 1.5h | |
| - Integration | 1h | |
| - Configuration | 0.25h | |
| - Testing | 1.5h | |
| **Day 4 Subtotal** | **4.25h** | |
| | | |
| **Total Week 2** | **~9h** | |

---

## File Structure After Week 2

```
backend/
├── src/
│   ├── services/
│   │   ├── retrieval_evaluator.py       [Week 1]
│   │   ├── query_refiner.py             [NEW - Day 3]
│   │   ├── clarification_helper.py      [NEW - Day 4]
│   │   ├── agent_tools.py               [MODIFIED]
│   │   └── agent_service.py             [No changes]
│   │
│   ├── models/
│   │   └── evaluation.py                [MODIFIED - add refinement fields]
│   │
│   └── config/
│       └── settings.py                  [MODIFIED - add refinement config]
│
├── tests/
│   ├── test_retrieval_evaluator_integration.py  [Week 1]
│   ├── test_query_refiner.py                    [NEW - Day 3]
│   └── test_week2_integration.py                [NEW - Day 4]
│
├── docs/
│   ├── WEEK2_OVERVIEW.md                [Planning]
│   ├── WEEK2_DAY3_IMPLEMENTATION.md     [Detailed guide]
│   ├── WEEK2_DAY4_IMPLEMENTATION.md     [Detailed guide]
│   └── WEEK2_CHECKLIST.md               [This file]
│
└── .env                                 [MODIFIED - add refinement settings]
```

---

## Next Steps After Week 2

### Week 3: External Search Integration (3-4 days)

- Integrate MCP (Model Context Protocol) tools
- Handle EXTERNAL recommendation
- Brave Search API integration
- Merge internal + external results
- Source attribution

---

## Documentation References

- **Overview**: [WEEK2_OVERVIEW.md](./WEEK2_OVERVIEW.md)
- **Day 3 Guide**: [WEEK2_DAY3_IMPLEMENTATION.md](./WEEK2_DAY3_IMPLEMENTATION.md)
- **Day 4 Guide**: [WEEK2_DAY4_IMPLEMENTATION.md](./WEEK2_DAY4_IMPLEMENTATION.md)
- **Week 1 Complete**: [DAY2_CHECKLIST.md](./DAY2_CHECKLIST.md)
- **Evaluation Models**: [../src/models/evaluation.py](../src/models/evaluation.py)

---

**Ready to Start Week 2?**

1. Read [WEEK2_OVERVIEW.md](./WEEK2_OVERVIEW.md) for big picture
2. Follow [WEEK2_DAY3_IMPLEMENTATION.md](./WEEK2_DAY3_IMPLEMENTATION.md) step-by-step
3. Use this checklist to track progress
4. Test thoroughly at each step

**Let's build intelligent query refinement! 🚀**
