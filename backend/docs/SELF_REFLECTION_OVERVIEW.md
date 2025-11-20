# Self-Reflection System Overview

Quick reference for the self-reflection (retrieval evaluation) implementation.

---

## What is Self-Reflection?

A system that **evaluates retrieval quality** before generating answers. It assesses whether retrieved contexts are good enough to answer the user's query, and recommends actions if not.

**Goal**: Improve answer quality by detecting when retrieval fails, rather than blindly answering with poor contexts.

---

## System Architecture

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
│ search_documents     │  ← Retrieves contexts from ChromaDB
└──────┬───────────────┘
       │
       ▼
┌──────────────────────────────┐
│ RetrievalEvaluator.evaluate()│  ← NEW: Self-reflection
└──────┬───────────────────────┘
       │
       ▼
┌─────────────────────┐
│ EvaluationResult    │
│ - quality           │
│ - confidence        │
│ - recommendation    │
└──────┬──────────────┘
       │
       ▼
  ┌────────────────────────────────┐
  │ Decision:                      │
  │ • ANSWER → use contexts        │
  │ • REFINE → reformulate query   │ (Week 2)
  │ • EXTERNAL → search web        │ (Week 3)
  │ • CLARIFY → ask user           │ (Week 2)
  └────────────────────────────────┘
```

---

## Three Evaluation Modes

| Mode | Latency | Cost | LLM Calls | Accuracy | When to Use |
|------|---------|------|-----------|----------|-------------|
| **FAST** | ~50ms | Free | 0 | Good | High traffic, budget-conscious |
| **BALANCED** | ~500ms | Low | 1 (quick) | Better | **Recommended default** |
| **THOROUGH** | ~2s | Medium | 1-2 (detailed) | Best | Critical queries only |

**Configuration**: Set via `REFLECTION_MODE` env variable (fast/balanced/thorough)

---

## Implementation Phases

### ✅ Week 1 (Day 1-2): Pure Evaluation
- **Day 1**: Data models (evaluation.py) - ✅ COMPLETE
- **Day 2**: RetrievalEvaluator implementation
- **Behavior**: Evaluate and **log only** - don't change agent behavior yet
- **Output**: Confidence scores, quality levels, recommendations (logged)

### 📅 Week 2 (Day 3-4): Action Taking - Query Refinement
- Implement automatic query refinement
- When evaluation returns REFINE → reformulate query → retry search
- Add refinement tracking (prevent infinite loops)

### 📅 Week 3 (Day 5-7): External Search Integration
- Integrate with MCP (Model Context Protocol) tools
- When evaluation returns EXTERNAL → trigger Brave Search
- Merge internal + external results

---

## Key Files

```
backend/src/
├── models/
│   └── evaluation.py                    ✅ Complete (Day 1)
│       - QualityLevel enum
│       - RecommendationAction enum
│       - ReflectionMode enum
│       - EvaluationCriteria dataclass
│       - EvaluationResult dataclass
│       - ReflectionConfig dataclass
│
├── services/
│   ├── retrieval_evaluator.py          🔨 TODO (Day 2)
│   │   - RetrievalEvaluator class
│   │   - _evaluate_fast()
│   │   - _evaluate_balanced()
│   │   - _evaluate_thorough()
│   │
│   ├── agent_tools.py                   🔧 Modify (Day 2)
│   │   - Add evaluation after search
│   │   - Store eval result in context
│   │
│   └── agent_service.py                 📝 No changes yet (Week 2)
│
└── config/
    └── settings.py                      ✅ Complete (Day 1)
        - USE_SELF_REFLECTION
        - REFLECTION_MODE
        - REFLECTION_THRESHOLD_*
        - REFLECTION_AUTO_*
```

---

## Configuration Reference

### Environment Variables (.env)

```bash
# Master switch
USE_SELF_REFLECTION=false              # Enable/disable entire feature

# Performance mode
REFLECTION_MODE=balanced               # fast | balanced | thorough

# Quality thresholds (0.0-1.0)
REFLECTION_THRESHOLD_EXCELLENT=0.85   # >= 0.85 → EXCELLENT
REFLECTION_THRESHOLD_GOOD=0.70        # >= 0.70 → GOOD
REFLECTION_THRESHOLD_PARTIAL=0.50     # >= 0.50 → PARTIAL, else POOR

# Behavior settings
REFLECTION_MIN_CONTEXTS=1             # Min contexts needed to answer
REFLECTION_AUTO_REFINE=true           # Auto-refine on poor quality (Week 2)
REFLECTION_AUTO_EXTERNAL=false        # Auto-search external (Week 3)
REFLECTION_MAX_REFINEMENT_ATTEMPTS=3  # Prevent infinite loops
```

### Python Usage

```python
from src.models.evaluation import ReflectionConfig, EvaluationCriteria
from src.services.retrieval_evaluator import RetrievalEvaluator
from src.config.settings import Config

# Load config from environment
config = ReflectionConfig.from_settings(Config)

# Create evaluator
evaluator = RetrievalEvaluator(
    config=config,
    openai_client=openai_client  # Required for BALANCED/THOROUGH
)

# Evaluate retrieval
criteria = EvaluationCriteria(
    query="What was Q1 revenue?",
    contexts=retrieved_contexts,
    search_metadata={"hybrid": True, "top_k": 5}
)

result = evaluator.evaluate(criteria)

# Check result
print(f"Quality: {result.quality.value}")           # excellent/good/partial/poor
print(f"Confidence: {result.confidence:.2f}")       # 0.0-1.0
print(f"Recommendation: {result.recommendation.value}")  # answer/refine/external/clarify
print(f"Reasoning: {result.reasoning}")

# Use properties for decision making
if result.should_answer:
    # Contexts are good, proceed with answer
    pass
elif result.should_refine:
    # Refine query and retry (Week 2)
    pass
elif result.should_search_external:
    # Trigger external search (Week 3)
    pass
```

---

## Evaluation Metrics

### Confidence Score (0.0-1.0)

Calculated differently per mode:

**FAST Mode** (heuristic formula):
```
confidence = (
    keyword_overlap * 0.4 +      # 40%: semantic relevance
    avg_score * 0.3 +             # 30%: retrieval scores
    min_score * 0.2 +             # 20%: worst-case quality
    context_presence * 0.1        # 10%: has contexts?
)
```

**BALANCED Mode**: Heuristic + LLM validation adjustment (±0.1)

**THOROUGH Mode**: LLM directly provides confidence score

### Quality Levels

Maps confidence to quality:
- **EXCELLENT**: confidence ≥ 0.85
- **GOOD**: confidence ≥ 0.70
- **PARTIAL**: confidence ≥ 0.50
- **POOR**: confidence < 0.50

### Recommendations

Decision logic:
- **Confidence ≥ 0.70** → ANSWER (contexts sufficient)
- **0.50 ≤ Confidence < 0.70** → REFINE (try better query)
- **Confidence < 0.50 + no contexts** → EXTERNAL (need other sources)
- **Confidence < 0.50 + has contexts** → CLARIFY (query ambiguous)

---

## Testing Checklist

### Manual Testing

```bash
# Test 1: Good retrieval (expect ANSWER)
curl -X POST http://localhost:5000/chat/agent \
  -d '{"message": "What is our vacation policy?", "collection_name": "hr_docs"}'

# Expected output in logs:
# [REFLECTION] Quality: good, Confidence: 0.82, Recommendation: answer

# Test 2: Poor retrieval (expect REFINE or EXTERNAL)
curl -X POST http://localhost:5000/chat/agent \
  -d '{"message": "What is the weather?", "collection_name": "hr_docs"}'

# Expected output in logs:
# [REFLECTION] Quality: poor, Confidence: 0.12, Recommendation: external
```

### Unit Testing

```python
# Test data models
python -m src.models.evaluation

# Test evaluator (once implemented)
pytest tests/test_retrieval_evaluator.py -v
```

---

## Troubleshooting

### Issue: Evaluation not running

**Check**:
1. `USE_SELF_REFLECTION=true` in .env
2. Config loaded correctly: `print(Config.USE_SELF_REFLECTION)`
3. Add debug prints in execute_search_documents()

### Issue: LLM calls failing in BALANCED/THOROUGH mode

**Check**:
1. OpenAI client passed in context: `context["openai_client"]`
2. API key is valid: `OPENAI_API_KEY` in .env
3. Check error logs for API errors

### Issue: Confidence scores seem wrong

**Check**:
1. Print intermediate metrics: keyword_overlap, avg_score, etc.
2. Verify threshold settings match expectations
3. Test with known good/bad queries
4. Adjust heuristic weights if needed

### Issue: All evaluations return same recommendation

**Check**:
1. Thresholds may be too high/low
2. Keyword overlap calculation may be broken
3. Test with diverse queries (good/bad/ambiguous)

---

## Performance Benchmarks

### Expected Latency

| Mode | No Reflection | FAST | BALANCED | THOROUGH |
|------|---------------|------|----------|----------|
| Search only | 150ms | - | - | - |
| + Evaluation | - | +50ms | +500ms | +2000ms |
| **Total** | **150ms** | **200ms** | **650ms** | **2150ms** |

### Cost Estimates (per query)

| Mode | LLM Calls | Tokens | Cost (GPT-4o-mini) |
|------|-----------|--------|---------------------|
| FAST | 0 | 0 | $0.000 |
| BALANCED | 1 quick | ~500 | $0.0001 |
| THOROUGH | 1-2 detailed | ~2000 | $0.0004 |

**Note**: These are evaluation costs only. Agent still makes its normal LLM calls for answer generation.

---

## Future Enhancements

### Week 4+: Advanced Features
- **Query understanding**: Analyze query intent, complexity
- **Multi-hop reasoning**: Break complex queries into sub-queries
- **Hybrid retrieval strategies**: Dense + sparse + graph
- **Feedback loops**: Learn from user corrections
- **A/B testing**: Compare with/without self-reflection

### Monitoring & Analytics
- Track evaluation metrics over time
- Identify common failure patterns
- Measure impact on answer quality
- Cost vs quality tradeoff analysis

---

## References

- Implementation Guide: [DAY2_IMPLEMENTATION_GUIDE.md](./DAY2_IMPLEMENTATION_GUIDE.md)
- Agentic RAG Roadmap: [C:/temp/Agentic_RAG_Guideline.md](C:/temp/Agentic_RAG_Guideline.md)
- Data Models: [src/models/evaluation.py](../src/models/evaluation.py)
- Configuration: [src/config/settings.py](../src/config/settings.py)

---

**Current Status**: Day 1 Complete ✅ | Day 2 In Progress 🔨

**Next Action**: Implement `retrieval_evaluator.py` following Day 2 guide
