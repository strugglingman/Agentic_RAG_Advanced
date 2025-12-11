# Week 2: Action-Taking - Query Refinement System

**Status**: 📋 Planning Phase
**Estimated Time**: 2-3 days
**Prerequisite**: Week 1 Complete ✅

---

## 🎯 Goals

Implement **automatic query refinement** when retrieval evaluation detects poor quality results.

**Week 1 Recap**: System evaluates retrieval quality and logs recommendations
**Week 2 Goal**: System **takes action** on REFINE/CLARIFY recommendations

---

## 📊 Architecture Overview

```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Agent (ReAct Loop)  │
└──────┬──────────────┘
       │
       ▼
┌──────────────────────┐
│ search_documents     │ ← Retrieves contexts
└──────┬───────────────┘
       │
       ▼
┌──────────────────────────────┐
│ RetrievalEvaluator.evaluate()│ ← Evaluates quality
└──────┬───────────────────────┘
       │
       ▼
  ┌────────────────────────────────────┐
  │ Decision Logic (NEW):              │
  │                                    │
  │ • ANSWER → return contexts         │ ✅ Already works
  │ • REFINE → reformulate + retry     │ ⬅ Week 2: Implement this
  │ • CLARIFY → ask user for details   │ ⬅ Week 2: Implement this
  │ • EXTERNAL → search web (Week 3)   │ 📅 Future
  └────────────────────────────────────┘
```

---

## 🎬 User Experience Flow

### Scenario 1: Good Retrieval (No Change)
```
User: "What is our vacation policy?"
  ↓
Agent searches documents → finds relevant contexts
  ↓
Evaluator: Quality=GOOD, Recommendation=ANSWER
  ↓
Agent returns answer ✅
```

### Scenario 2: Poor Retrieval → Query Refinement (NEW)
```
User: "Tell me about PTO"
  ↓
Agent searches documents → finds weak contexts (low confidence)
  ↓
Evaluator: Quality=PARTIAL, Recommendation=REFINE
  ↓
Agent reformulates query → "employee paid time off vacation policy"
  ↓
Agent retries search → finds better contexts
  ↓
Evaluator: Quality=GOOD, Recommendation=ANSWER
  ↓
Agent returns answer ✅
```

### Scenario 3: Ambiguous Query → Clarification (NEW)
```
User: "How do I apply?"
  ↓
Agent searches documents → finds no relevant contexts
  ↓
Evaluator: Quality=POOR, Recommendation=CLARIFY
  ↓
Agent asks: "I found no relevant information. Could you clarify what you'd like to apply for? (e.g., job application, leave request, benefits enrollment)"
  ↓
User provides clarification → restart with refined query ✅
```

---

## 🔑 Key Features to Implement

### 1. **Query Refinement Logic** (Priority 1)
- Detect REFINE recommendation in agent_tools.py
- Use LLM to reformulate query based on:
  - Original query
  - Evaluation issues (e.g., "poor keyword match", "missing context")
  - Missing aspects (e.g., keywords not found)
- Retry search with refined query
- Track refinement attempts (max 3 to prevent loops)

### 2. **Clarification Request Logic** (Priority 2)
- Detect CLARIFY recommendation
- Generate helpful clarification message based on:
  - Why query failed (missing context, ambiguous, out of scope)
  - Suggestions for what user might mean
- Return clarification to user (don't retry automatically)

### 3. **Refinement State Tracking** (Priority 3)
- Add to context: `_refinement_count`, `_original_query`, `_refinement_history`
- Prevent infinite loops (max 3 refinements)
- Progressive fallback: REFINE → REFINE → REFINE → EXTERNAL/CLARIFY

### 4. **Configuration & Controls** (Priority 4)
- Add env vars: `REFLECTION_AUTO_REFINE`, `REFLECTION_MAX_REFINEMENT_ATTEMPTS`
- Allow disabling auto-refinement (default: enabled)
- Allow per-request override via API parameter

---

## 🏗️ Implementation Phases

### **Day 3 (4-5 hours): Core Refinement Logic**

#### Part 1: Query Refinement Service
- Create `src/services/query_refiner.py`
- Implement `QueryRefiner` class with:
  - `refine_query(original_query, eval_result) -> refined_query`
  - Uses LLM with prompt template
  - Incorporates issues and missing_aspects from evaluation

#### Part 2: State Tracking
- Add refinement tracking to context dict
- Track: attempt count, original query, refinement history
- Implement max attempts logic

#### Part 3: Integration with agent_tools.py
- Modify `execute_search_documents()` to:
  - Check if refinement is needed (eval_result.should_refine)
  - Call QueryRefiner if needed
  - Retry search with refined query
  - Update context tracking

### **Day 4 (3-4 hours): Clarification & Testing**

#### Part 4: Clarification Logic
- Implement clarification message generation
- Add to `execute_search_documents()`:
  - Detect CLARIFY recommendation
  - Generate helpful clarification message
  - Return to user (no retry)

#### Part 5: Progressive Fallback
- Implement escalation strategy:
  - Attempt 1-2: REFINE
  - Attempt 3: EXTERNAL (if enabled) or CLARIFY
- Add fallback logic to handle max refinements

#### Part 6: Testing & Validation
- Test refinement scenarios
- Test clarification scenarios
- Test loop prevention (max attempts)
- Test progressive fallback

---

## 📁 Files to Modify/Create

### New Files:
```
backend/src/services/
├── query_refiner.py          [NEW] Query reformulation logic
└── clarification_helper.py   [NEW] (Optional) Clarification message generation
```

### Modified Files:
```
backend/src/services/
├── agent_tools.py            [MODIFY] Add refinement decision logic
└── agent_service.py          [MODIFY] (Maybe) Pass refinement context

backend/src/config/
└── settings.py               [MODIFY] Add refinement config vars

backend/.env                   [MODIFY] Add refinement settings
```

### New Tests:
```
backend/tests/
├── test_query_refiner.py              [NEW] Unit tests for refinement
└── test_refinement_integration.py     [NEW] End-to-end refinement tests
```

---

## 🎚️ Configuration Settings

### New Environment Variables:

```bash
# Query Refinement Settings
REFLECTION_AUTO_REFINE=true                  # Enable automatic refinement
REFLECTION_MAX_REFINEMENT_ATTEMPTS=3         # Max retry attempts
REFLECTION_REFINEMENT_PROMPT_TEMPLATE=default # Prompt template to use

# Clarification Settings
REFLECTION_AUTO_CLARIFY=true                 # Enable clarification messages
REFLECTION_CLARIFY_WITH_SUGGESTIONS=true     # Include suggestions in clarification
```

### Runtime Config (ReflectionConfig):
```python
@dataclass
class ReflectionConfig:
    # ... existing fields ...

    # Week 2: Refinement settings
    auto_refine: bool = True
    max_refinement_attempts: int = 3
    auto_clarify: bool = True
    refinement_prompt_template: str = "default"
```

---

## 🧪 Testing Strategy

### Unit Tests:
1. **QueryRefiner tests**:
   - Test query reformulation with various issues
   - Test missing aspects incorporation
   - Test prompt template variations

2. **State tracking tests**:
   - Test refinement count tracking
   - Test max attempts enforcement
   - Test refinement history logging

### Integration Tests:
1. **End-to-end refinement**:
   - Test poor query → refinement → success
   - Test multiple refinements (progressive)
   - Test max attempts → fallback to clarify

2. **Clarification scenarios**:
   - Test ambiguous query → clarification request
   - Test no contexts → helpful clarification

3. **Loop prevention**:
   - Test max attempts prevents infinite loops
   - Test fallback to CLARIFY after max attempts

---

## 🎯 Success Criteria

Week 2 is complete when:

- [ ] QueryRefiner service implemented and tested
- [ ] Refinement logic integrated into agent_tools.py
- [ ] Refinement tracking prevents infinite loops (max 3 attempts)
- [ ] Clarification messages generated for ambiguous queries
- [ ] Progressive fallback works: REFINE → REFINE → CLARIFY
- [ ] All tests passing
- [ ] Manual testing with real queries validates behavior
- [ ] Logs show refinement attempts and outcomes
- [ ] Configuration allows enabling/disabling refinement

---

## 🚀 Next Steps After Week 2

### Week 3: External Search Integration
- Integrate MCP (Model Context Protocol) tools
- Implement EXTERNAL recommendation handling
- Brave Search API integration
- Merge internal + external results

---

## 📚 References

- Week 1 Implementation: [DAY2_CHECKLIST.md](./DAY2_CHECKLIST.md)
- Evaluation Models: [../src/models/evaluation.py](../src/models/evaluation.py)
- Current Agent Tools: [../src/services/agent_tools.py](../src/services/agent_tools.py)
- RetrievalEvaluator: [../src/services/retrieval_evaluator.py](../src/services/retrieval_evaluator.py)

---

**Next Action**: Read [WEEK2_DAY3_IMPLEMENTATION.md](./WEEK2_DAY3_IMPLEMENTATION.md) for detailed implementation steps.
