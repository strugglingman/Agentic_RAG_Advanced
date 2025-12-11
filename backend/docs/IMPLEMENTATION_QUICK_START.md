# Self-Reflection Implementation Quick Start

This guide shows you exactly where to start and what files to implement.

---

## Current Status

✅ **Day 1 Complete**:
- `src/models/evaluation.py` - All data models implemented and tested
- `src/config/settings.py` - Configuration parameters added
- `.env` - Environment variables configured

✅ **Day 2 Complete**:
- `src/services/retrieval_evaluator.py` - Fully implemented with all 3 modes (FAST, BALANCED, THOROUGH)
- `src/services/agent_tools.py` - Integration complete with logging
- `src/routes/chat.py` - OpenAI client passed to agent context
- All unit tests passing
- Integration tests passing

---

## Files Created with TODOs

### 1. **Main Implementation File** ⭐ START HERE
**File**: `d:\chatbot\backend\src\services\retrieval_evaluator.py`

**Status**: Skeleton created with detailed TODO comments

**What to implement**:
```python
class RetrievalEvaluator:
    __init__()           # TODO: Initialize config and client
    evaluate()           # TODO: Route to appropriate mode
    _evaluate_fast()     # TODO: Implement heuristic evaluation (MOST IMPORTANT)
    _evaluate_balanced() # TODO: Implement heuristics + LLM
    _evaluate_thorough() # TODO: Implement full LLM evaluation

    # Helper methods (all have TODOs):
    _extract_keywords()
    _extract_context_text()
    _calculate_keyword_overlap()
    _detect_issues()
    _identify_missing_aspects()
    _determine_recommendation()
    _extract_relevance_scores()
    _quick_llm_check()
```

**Implementation Order** (recommended):
1. Start with helper methods (bottom-up)
2. Then implement `_evaluate_fast()` (core heuristic logic)
3. Test FAST mode thoroughly
4. Implement `_evaluate_balanced()` and `_quick_llm_check()`
5. Test BALANCED mode
6. Implement `_evaluate_thorough()`
7. Test THOROUGH mode

**Test Command**:
```bash
cd backend
python -m src.services.retrieval_evaluator
```

---

### 2. **Integration File #1**
**File**: `d:\chatbot\backend\src\services\agent_tools.py`

**Status**: TODO block added at line ~121-187

**What to implement**:
After `context["_retrieved_contexts"].extend(ctx)`, add the self-reflection evaluation logic:

```python
# Check if Config.USE_SELF_REFLECTION is True
if Config.USE_SELF_REFLECTION:
    # Import evaluator classes
    # Load config
    # Create evaluator
    # Build evaluation criteria
    # Run evaluation
    # Store result in context
    # Log evaluation result
```

**Full implementation steps** are documented in the TODO block in the file.

---

### 3. **Integration File #2**
**File**: `d:\chatbot\backend\src\routes\chat.py`

**Status**: TODO block added at line ~220-240

**What to implement**:
Uncomment this line in `agent_context` dict:
```python
"openai_client": openai_client,
```

This passes the OpenAI client to the agent tools so the evaluator can make LLM calls.

---

## Implementation Steps (Detailed)

### Step 1: Implement Helper Methods (30 min)

Open `src/services/retrieval_evaluator.py` and implement these methods first:

1. **`_extract_keywords()`** - Extract keywords from query
   - Remove stopwords
   - Lowercase and split
   - Return list of keywords

2. **`_extract_context_text()`** - Combine all context text
   - Get "chunk" field from each context
   - Join with spaces
   - Return lowercase string

3. **`_calculate_keyword_overlap()`** - Calculate overlap score
   - Count keywords found in context text
   - Return ratio (0.0-1.0)

4. **`_detect_issues()`** - Find problems with retrieval
   - Check context count, scores, overlap
   - Return list of issue strings

5. **`_identify_missing_aspects()`** - Find missing keywords
   - Filter keywords not in context text
   - Return list

6. **`_determine_recommendation()`** - Decide action
   - Use confidence thresholds
   - Return (RecommendationAction, reasoning)

7. **`_extract_relevance_scores()`** - Get scores from contexts
   - Extract "score" or "distance" fields
   - Return list of floats

**Test**: Run the test block to verify helpers work

---

### Step 2: Implement `__init__()` and `evaluate()` (15 min)

8. **`__init__()`** - Initialize evaluator
   - Store config and client
   - Validate client for BALANCED/THOROUGH modes

9. **`evaluate()`** - Route to appropriate method
   - Get mode from criteria
   - Call _evaluate_fast/balanced/thorough

**Test**: Create evaluator and call evaluate() with FAST mode

---

### Step 3: Implement FAST Mode (1-2 hours) ⭐ KEY

10. **`_evaluate_fast()`** - Core heuristic evaluation
    - Extract metrics (context count, keywords, scores)
    - Calculate confidence using weighted formula
    - Calculate coverage
    - Detect issues and missing aspects
    - Determine recommendation
    - Build and return EvaluationResult

**This is the most important method!** All three modes depend on this working correctly.

**Test extensively**:
```python
# Test with good contexts (expect high confidence, ANSWER)
# Test with poor contexts (expect low confidence, REFINE/EXTERNAL)
# Test with no contexts (expect EXTERNAL)
# Test with borderline contexts (expect PARTIAL quality)
```

---

### Step 4: Implement BALANCED Mode (1 hour)

11. **`_evaluate_balanced()`** - Heuristics + LLM validation
    - Call `_evaluate_fast()` first
    - If confidence is borderline, call `_quick_llm_check()`
    - Adjust confidence based on LLM response
    - Update mode_used

12. **`_quick_llm_check()`** - Light LLM validation
    - Format contexts for LLM
    - Build prompt asking "Can these answer the query?"
    - Call OpenAI API (gpt-4o-mini)
    - Parse response (yes/partial/no)
    - Adjust confidence accordingly

**Test**: Enable BALANCED mode and verify LLM calls work

---

### Step 5: Implement THOROUGH Mode (1-2 hours)

13. **`_evaluate_thorough()`** - Full LLM evaluation
    - Format contexts with full details
    - Build comprehensive evaluation prompt
    - Call OpenAI API requesting JSON response
    - Parse LLM JSON output
    - Build EvaluationResult from LLM data
    - Fallback to BALANCED mode on error

**Test**: Enable THOROUGH mode and verify detailed evaluation

---

### Step 6: Integration (30 min)

14. **In `agent_tools.py`**: Implement the TODO block
    - Add imports
    - Check USE_SELF_REFLECTION flag
    - Create evaluator
    - Build criteria
    - Run evaluation
    - Log results

15. **In `chat.py`**: Uncomment OpenAI client line
    - Uncomment: `"openai_client": openai_client,`

**Test**: Run full agent chat with self-reflection enabled

---

### Step 7: End-to-End Testing (30 min)

16. **Enable self-reflection** in `.env`:
    ```bash
    USE_SELF_REFLECTION=true
    REFLECTION_MODE=fast
    ```

17. **Restart backend server**:
    ```bash
    cd backend
    python app.py
    ```

18. **Test with good query**:
    ```bash
    curl -X POST http://localhost:5000/chat/agent \
      -H "Content-Type: application/json" \
      -d '{"message": "What is our vacation policy?", "collection_name": "hr_docs"}'
    ```

    **Expected logs**:
    ```
    [SELF-REFLECTION] Quality: good, Confidence: 0.82, Recommendation: answer
    [SELF-REFLECTION] Reasoning: Good confidence - contexts provide sufficient information
    ```

19. **Test with bad query**:
    ```bash
    curl -X POST http://localhost:5000/chat/agent \
      -H "Content-Type: application/json" \
      -d '{"message": "What is quantum physics?", "collection_name": "hr_docs"}'
    ```

    **Expected logs**:
    ```
    [SELF-REFLECTION] Quality: poor, Confidence: 0.15, Recommendation: external
    [SELF-REFLECTION] Reasoning: No contexts found - may need external search
    [SELF-REFLECTION] Issues: No contexts retrieved
    ```

20. **Test BALANCED mode**:
    - Change `.env`: `REFLECTION_MODE=balanced`
    - Restart server
    - Repeat tests
    - Verify LLM validation runs for borderline cases

21. **Test THOROUGH mode**:
    - Change `.env`: `REFLECTION_MODE=thorough`
    - Restart server
    - Repeat tests
    - Verify detailed LLM evaluation runs

---

## Verification Checklist

Day 2 - **ALL COMPLETE** ✅:

- [x] All helper methods implemented in `retrieval_evaluator.py`
- [x] `__init__()` and `evaluate()` routing work correctly
- [x] FAST mode implemented and tested
- [x] BALANCED mode implemented and tested
- [x] THOROUGH mode implemented and tested
- [x] Integration in `agent_tools.py` complete
- [x] OpenAI client passed in `chat.py`
- [x] Unit tests pass (python -m src.models.evaluation)
- [x] Integration tests pass (python -m src.services.retrieval_evaluator)
- [x] End-to-end tests pass (test_self_reflection.py)
- [x] Evaluation logs appear correctly
- [x] Agent behavior unchanged (Week 1: evaluation only)
- [x] No errors or crashes

---

## Time Estimates

- **Step 1** (Helpers): 30 minutes
- **Step 2** (Init/Route): 15 minutes
- **Step 3** (FAST mode): 1-2 hours ⭐
- **Step 4** (BALANCED mode): 1 hour
- **Step 5** (THOROUGH mode): 1-2 hours
- **Step 6** (Integration): 30 minutes
- **Step 7** (Testing): 30 minutes

**Total**: 5-7 hours (focused work, full day)

---

## Common Issues & Solutions

### Issue: Import errors
```
ModuleNotFoundError: No module named 'src.services.retrieval_evaluator'
```
**Solution**: Make sure file exists and restart server

### Issue: OpenAI client is None
```
ValueError: OpenAI client required for BALANCED/THOROUGH modes
```
**Solution**: Uncomment `"openai_client": openai_client` in chat.py

### Issue: Evaluation not running
**Solution**: Check `USE_SELF_REFLECTION=true` in .env and restart server

### Issue: Confidence always 0 or 1
**Solution**: Debug keyword_overlap and score extraction, add print statements

### Issue: LLM calls timing out
**Solution**: Increase timeout or use faster model (gpt-3.5-turbo)

---

## Next Actions

**✅ Day 2 Complete!**

The self-reflection system is now fully operational:
- All evaluation modes working (FAST, BALANCED, THOROUGH)
- Integration complete in agent tools
- Comprehensive logging implemented
- All tests passing

**Quick Test**:
```bash
cd backend
python test_self_reflection.py
```

**What's Next - Week 2+**:
- Week 2: Implement action-taking (query refinement based on evaluation)
- Week 3: Integrate external search (MCP) when recommendation is EXTERNAL
- Week 4+: Add analytics, monitoring, and performance optimization

---

## Reference Documentation

- **Detailed Implementation**: [DAY2_IMPLEMENTATION_GUIDE.md](./DAY2_IMPLEMENTATION_GUIDE.md)
- **Step-by-Step Checklist**: [DAY2_CHECKLIST.md](./DAY2_CHECKLIST.md)
- **System Overview**: [SELF_REFLECTION_OVERVIEW.md](./SELF_REFLECTION_OVERVIEW.md)

---

**Ready to code? Start with `retrieval_evaluator.py` Step 1! 🚀**
