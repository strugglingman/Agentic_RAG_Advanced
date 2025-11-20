# Day 2: RetrievalEvaluator Implementation & Integration

**Goal**: Implement the core evaluation logic and integrate it into the agent service to evaluate retrieval quality in real-time.

**Status**: Data models complete ✓ | Evaluator implementation (TODO) | Integration (TODO)

---

## Overview

Today you'll create the `RetrievalEvaluator` class that uses the data models from Day 1 to evaluate whether retrieved contexts are good enough to answer a user's query.

### What We're Building

```
User Query → search_documents tool → Retrieved Contexts
                                              ↓
                                    RetrievalEvaluator.evaluate()
                                              ↓
                                    EvaluationResult (quality + recommendation)
                                              ↓
                        Decision: ANSWER / REFINE / EXTERNAL / CLARIFY
```

---

## Architecture Decision

The evaluator will support **three modes** with different speed/accuracy tradeoffs:

1. **FAST Mode** (~50ms, no LLM calls)
   - Heuristics only: keyword overlap, context count, score thresholds
   - Cheap, fast, good enough for most cases
   - Use when: high traffic, budget-conscious, acceptable accuracy

2. **BALANCED Mode** (~500ms, 1 lightweight LLM call)
   - Heuristics + quick LLM assessment
   - Best of both worlds: reasonably fast + intelligent
   - **RECOMMENDED** as default
   - Use when: need better accuracy without high latency

3. **THOROUGH Mode** (~2s, 1-2 detailed LLM calls)
   - Full LLM-based evaluation with detailed reasoning
   - Highest accuracy, most expensive
   - Use when: critical queries, willing to pay for quality

---

## Part 1: File Structure

Create the following file:

```
backend/src/services/
├── agent_service.py          # Existing - will integrate evaluator here
├── agent_tools.py            # Existing - will add evaluation after search
├── retrieval.py              # Existing - unchanged
└── retrieval_evaluator.py    # NEW - implement today
```

---

## Part 2: RetrievalEvaluator Class Design

### File: `src/services/retrieval_evaluator.py`

#### TODO 1: Create the class skeleton

```python
"""
Retrieval quality evaluator for the self-reflection system.

This module implements the RetrievalEvaluator class which assesses whether
retrieved contexts are sufficient to answer a user's query.

Supports three evaluation modes:
- FAST: Heuristic-based evaluation (~50ms)
- BALANCED: Heuristics + light LLM check (~500ms)
- THOROUGH: Full LLM evaluation (~2s)
"""

# What imports do you need?
# - OpenAI client for LLM calls
# - Data models from evaluation.py
# - Config for settings
# - typing for type hints

class RetrievalEvaluator:
    """
    Evaluates retrieval quality and recommends next actions.

    Attributes:
        config: ReflectionConfig instance
        openai_client: OpenAI client for LLM-based evaluation
    """

    def __init__(self, config: ReflectionConfig, openai_client: Optional[OpenAI] = None):
        """
        Initialize the evaluator.

        Args:
            config: Runtime configuration for evaluation
            openai_client: Optional OpenAI client (required for BALANCED/THOROUGH modes)

        TODO:
        1. Store config
        2. Store openai_client
        3. Validate: if mode is BALANCED or THOROUGH, openai_client must not be None
        """
        pass

    def evaluate(self, criteria: EvaluationCriteria) -> EvaluationResult:
        """
        Evaluate retrieval quality.

        Args:
            criteria: Input containing query, contexts, metadata

        Returns:
            EvaluationResult with quality assessment + recommendation

        TODO:
        1. Route to appropriate evaluation method based on config.mode
        2. Return EvaluationResult
        """
        pass
```

---

## Part 3: Implement FAST Mode (Heuristic-Based)

**FAST mode** uses only heuristics - no LLM calls. This is the fastest and cheapest option.

### TODO 2: Implement `_evaluate_fast()`

This method should calculate:

#### Step 2.1: Calculate Basic Metrics

```python
def _evaluate_fast(self, criteria: EvaluationCriteria) -> EvaluationResult:
    """
    Fast heuristic-based evaluation.

    Metrics to calculate:
    1. context_count: len(criteria.contexts)
    2. keyword_overlap: measure query keywords in contexts
    3. avg_score: average of context scores from metadata
    4. min_score: minimum context score

    TODO: Extract these metrics from criteria
    """
```

#### Step 2.2: Calculate Keyword Overlap Score

```python
# How to calculate keyword_overlap:
# 1. Extract keywords from query (lowercase, split, remove stopwords)
# 2. Extract text from all contexts
# 3. Calculate: (keywords found in contexts) / (total keywords)
# 4. Return score 0.0-1.0

# Stopwords to ignore: "the", "a", "an", "is", "are", "was", "were", "what", "when", "where", "how", "why", "who"
```

#### Step 2.3: Calculate Confidence Score

Confidence formula (weighted average):
```python
confidence = (
    keyword_overlap * 0.4 +      # 40% weight: semantic relevance
    avg_score * 0.3 +             # 30% weight: retrieval scores
    min_score * 0.2 +             # 20% weight: worst-case quality
    context_presence * 0.1        # 10% weight: has contexts?
)

# Where:
# - keyword_overlap: 0.0-1.0 (from step 2.2)
# - avg_score: average of context scores
# - min_score: minimum context score
# - context_presence: 1.0 if len(contexts) >= config.min_contexts, else 0.0
```

#### Step 2.4: Calculate Coverage Score

```python
# Coverage: How well do contexts cover the query?
# Simple heuristic: keyword_overlap * context_quality_factor

coverage = keyword_overlap * min(1.0, avg_score * 1.2)

# Why this formula?
# - Higher keyword overlap = better coverage
# - Higher avg_score = contexts are more relevant
# - Cap at 1.0
```

#### Step 2.5: Detect Issues

```python
issues = []

# Check 1: No contexts retrieved?
if context_count == 0:
    issues.append("No contexts retrieved")

# Check 2: Too few contexts?
elif context_count < config.min_contexts:
    issues.append(f"Only {context_count} contexts found (min: {config.min_contexts})")

# Check 3: Low average score?
if avg_score < 0.5:
    issues.append(f"Low average relevance score: {avg_score:.2f}")

# Check 4: Poor keyword overlap?
if keyword_overlap < 0.3:
    issues.append(f"Low keyword overlap: {keyword_overlap:.2f}")
```

#### Step 2.6: Identify Missing Aspects

```python
# Simple heuristic: keywords not found in any context
missing_aspects = [keyword for keyword in query_keywords if keyword not in all_context_text]
```

#### Step 2.7: Determine Recommendation

```python
# Decision tree:
if confidence >= config.thresholds["excellent"]:
    recommendation = RecommendationAction.ANSWER
    reasoning = "High confidence - contexts directly answer the query"

elif confidence >= config.thresholds["good"]:
    recommendation = RecommendationAction.ANSWER
    reasoning = "Good confidence - contexts provide sufficient information"

elif context_count == 0:
    recommendation = RecommendationAction.EXTERNAL  # or REFINE
    reasoning = "No contexts found - may need external search"

elif confidence >= config.thresholds["partial"]:
    recommendation = RecommendationAction.REFINE
    reasoning = "Partial confidence - query refinement may help"

else:
    recommendation = RecommendationAction.CLARIFY
    reasoning = "Low confidence - query may be ambiguous or out of scope"
```

#### Step 2.8: Build and Return Result

```python
quality = config.get_quality_level(confidence)

return EvaluationResult(
    quality=quality,
    confidence=confidence,
    coverage=coverage,
    recommendation=recommendation,
    reasoning=reasoning,
    relevance_scores=[...],  # extract from contexts
    issues=issues,
    missing_aspects=missing_aspects,
    metrics={
        "keyword_overlap": keyword_overlap,
        "avg_score": avg_score,
        "min_score": min_score,
        "context_count": context_count,
    },
    mode_used=ReflectionMode.FAST,
)
```

---

## Part 4: Implement BALANCED Mode (Heuristics + Light LLM)

**BALANCED mode** runs heuristics first (FAST mode), then uses a lightweight LLM call to validate/adjust the assessment.

### TODO 3: Implement `_evaluate_balanced()`

```python
def _evaluate_balanced(self, criteria: EvaluationCriteria) -> EvaluationResult:
    """
    Balanced evaluation: heuristics + light LLM check.

    Steps:
    1. Run FAST mode first (get baseline assessment)
    2. If confidence is borderline (0.5-0.75), call LLM for validation
    3. Adjust confidence/recommendation based on LLM feedback
    4. Return updated result

    TODO:
    - Reuse _evaluate_fast() for baseline
    - Define "borderline" threshold (e.g., 0.5 < confidence < 0.75)
    - If borderline, call _quick_llm_check()
    - Adjust result based on LLM response
    - Update mode_used to BALANCED
    """
    pass
```

### TODO 4: Implement `_quick_llm_check()`

This is a lightweight LLM call to validate borderline cases.

```python
def _quick_llm_check(
    self, query: str, contexts: List[Dict[str, Any]], baseline_confidence: float
) -> Tuple[float, str]:
    """
    Quick LLM check for borderline cases.

    Args:
        query: User's question
        contexts: Retrieved contexts
        baseline_confidence: Confidence from heuristics

    Returns:
        Tuple of (adjusted_confidence, llm_reasoning)

    Prompt design:
    - Give LLM the query and contexts
    - Ask: "Can these contexts answer the query? (yes/partial/no)"
    - Parse response and adjust confidence accordingly

    TODO:
    1. Format contexts into readable text
    2. Build prompt (see template below)
    3. Call OpenAI API (use gpt-4o-mini for speed)
    4. Parse response (yes -> confidence += 0.1, no -> confidence -= 0.1)
    5. Return adjusted confidence + LLM reasoning
    """
    pass
```

#### Prompt Template for Quick LLM Check

```python
prompt = f"""You are evaluating whether retrieved contexts can answer a user's query.

USER QUERY: {query}

RETRIEVED CONTEXTS:
{formatted_contexts}

QUESTION: Can these contexts adequately answer the user's query?

Respond in this format:
ANSWER: [yes/partial/no]
REASONING: [one sentence explanation]

Be strict: only answer 'yes' if contexts directly and completely answer the query.
"""
```

---

## Part 5: Implement THOROUGH Mode (Full LLM Evaluation)

**THOROUGH mode** uses a detailed LLM prompt to comprehensively evaluate retrieval quality.

### TODO 5: Implement `_evaluate_thorough()`

```python
def _evaluate_thorough(self, criteria: EvaluationCriteria) -> EvaluationResult:
    """
    Thorough LLM-based evaluation.

    Steps:
    1. Format query + contexts for LLM
    2. Use detailed evaluation prompt
    3. Parse LLM response (structured output)
    4. Build EvaluationResult from LLM analysis

    TODO:
    - Build detailed prompt (see template below)
    - Call OpenAI API with JSON mode or structured output
    - Parse LLM response into EvaluationResult fields
    - Handle errors (fallback to BALANCED mode?)
    """
    pass
```

#### Prompt Template for Thorough Evaluation

```python
prompt = f"""You are an expert retrieval quality evaluator. Analyze whether the retrieved contexts can answer the user's query.

USER QUERY:
{query}

RETRIEVED CONTEXTS:
{formatted_contexts}

Evaluate the following:

1. RELEVANCE: Are contexts relevant to the query? (0.0-1.0 per context)
2. COVERAGE: Do contexts fully cover all aspects of the query? (0.0-1.0)
3. CONFIDENCE: Overall confidence that these contexts can answer the query? (0.0-1.0)
4. ISSUES: What problems exist? (list)
5. MISSING: What query aspects are not covered? (list)
6. RECOMMENDATION: What should we do?
   - ANSWER: Contexts are sufficient
   - REFINE: Reformulate query to get better results
   - EXTERNAL: Search external sources
   - CLARIFY: Query is ambiguous

Respond in JSON format:
{{
    "relevance_scores": [0.0-1.0, ...],
    "coverage": 0.0-1.0,
    "confidence": 0.0-1.0,
    "issues": ["issue1", "issue2"],
    "missing_aspects": ["aspect1", "aspect2"],
    "recommendation": "ANSWER|REFINE|EXTERNAL|CLARIFY",
    "reasoning": "one sentence explanation"
}}
"""
```

---

## Part 6: Integration with Agent Service

Now integrate the evaluator into your existing agent workflow.

### TODO 6: Modify `agent_tools.py` - Add Evaluation After Search

```python
# In execute_search_documents():

def execute_search_documents(args: Dict[str, Any], context: Dict[str, Any]) -> str:
    """Execute search with optional self-reflection evaluation."""

    # ... existing search logic ...
    results = retrieve(...)

    # NEW: Self-reflection evaluation (if enabled)
    if Config.USE_SELF_REFLECTION:
        from src.services.retrieval_evaluator import RetrievalEvaluator
        from src.models.evaluation import ReflectionConfig, EvaluationCriteria

        # Load config
        config = ReflectionConfig.from_settings(Config)

        # Create evaluator
        evaluator = RetrievalEvaluator(
            config=config,
            openai_client=context.get("openai_client")  # pass from context
        )

        # Build evaluation criteria
        criteria = EvaluationCriteria(
            query=query,
            contexts=results,  # retrieved contexts
            search_metadata={
                "hybrid": use_hybrid,
                "reranker": use_reranker,
                "top_k": top_k
            },
            mode=config.mode
        )

        # Evaluate
        eval_result = evaluator.evaluate(criteria)

        # Store evaluation in context for later use
        context["_last_evaluation"] = eval_result

        # Log evaluation (optional)
        print(f"[REFLECTION] Quality: {eval_result.quality.value}, "
              f"Confidence: {eval_result.confidence:.2f}, "
              f"Recommendation: {eval_result.recommendation.value}")

        # TODO Week 2+: Take action based on recommendation
        # For Week 1: Just log, don't change behavior yet

    # ... existing formatting logic ...
    return formatted_results
```

### TODO 7: Pass OpenAI Client in Context

Modify where the agent is created to pass the OpenAI client:

```python
# In your route handler (e.g., routes/chat.py or similar):

def chat_agent_handler():
    # ... existing code ...

    # Create context
    context = {
        "collection": collection,
        "dept_id": dept_id,
        "user_id": user_id,
        "openai_client": client,  # ADD THIS - pass OpenAI client
        "request_data": request_data,
    }

    # Run agent
    agent = Agent(openai_client=client, ...)
    answer, contexts = agent.run(query, context, messages_history)
```

---

## Part 7: Testing Strategy

### Test 1: Unit Test for Heuristic Functions

```python
# File: tests/test_retrieval_evaluator.py

def test_keyword_overlap():
    """Test keyword overlap calculation."""
    # TODO:
    # 1. Create test query: "What was Q1 revenue?"
    # 2. Create test contexts with various keyword coverage
    # 3. Calculate overlap
    # 4. Assert overlap is in expected range
    pass

def test_confidence_calculation():
    """Test confidence scoring."""
    # TODO:
    # 1. Create various scenarios (good/bad contexts)
    # 2. Calculate confidence
    # 3. Assert confidence matches expectations
    pass
```

### Test 2: Integration Test with Real Data

```python
def test_evaluate_fast_mode():
    """Test FAST mode evaluation end-to-end."""
    # TODO:
    # 1. Create ReflectionConfig with FAST mode
    # 2. Create EvaluationCriteria with sample data
    # 3. Create evaluator and run evaluate()
    # 4. Assert EvaluationResult fields are populated correctly
    pass

def test_evaluate_balanced_mode():
    """Test BALANCED mode with mock LLM."""
    # TODO:
    # 1. Mock OpenAI API responses
    # 2. Test borderline confidence cases
    # 3. Verify LLM is called when expected
    # 4. Verify confidence adjustment works
    pass
```

### Test 3: Manual Testing via API

```bash
# Test 1: Good query with relevant contexts
curl -X POST http://localhost:5000/chat/agent \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is our vacation policy?",
    "collection_name": "test_docs"
  }'

# Expected: High confidence, ANSWER recommendation

# Test 2: Query with no matches
curl -X POST http://localhost:5000/chat/agent \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the weather tomorrow?",
    "collection_name": "test_docs"
  }'

# Expected: Low confidence, EXTERNAL or CLARIFY recommendation
```

---

## Part 8: Success Criteria for Day 2

Before moving to Day 3, verify:

- [ ] `retrieval_evaluator.py` file created
- [ ] `RetrievalEvaluator` class implemented with all three modes
- [ ] FAST mode works (heuristics-based, no LLM)
- [ ] BALANCED mode works (heuristics + quick LLM check)
- [ ] THOROUGH mode works (detailed LLM evaluation)
- [ ] Integration with `agent_tools.py` complete
- [ ] Evaluation results are logged (visible in console/logs)
- [ ] OpenAI client properly passed through context
- [ ] Manual testing shows evaluation working for good/bad queries
- [ ] Week 1 behavior: Evaluation runs but doesn't change agent behavior yet

---

## Part 9: Common Pitfalls to Avoid

### Pitfall 1: LLM Client Not Available in BALANCED/THOROUGH Modes
**Solution**: Validate in `__init__()` that openai_client is provided for non-FAST modes

### Pitfall 2: Heuristic Weights Don't Match Reality
**Solution**: Start with suggested weights, then tune based on real data

### Pitfall 3: Prompt Engineering - LLM Returns Wrong Format
**Solution**: Use OpenAI JSON mode or function calling for structured output

### Pitfall 4: Evaluation Slows Down Every Query Too Much
**Solution**: Use FAST mode by default, only use BALANCED/THOROUGH for critical queries

### Pitfall 5: Forgetting to Update mode_used in Result
**Solution**: Always set `mode_used` field to the actual mode that ran

---

## Part 10: Debugging Checklist

If evaluation isn't working:

1. **Check config loading**
   ```python
   from src.models.evaluation import ReflectionConfig
   from src.config.settings import Config
   config = ReflectionConfig.from_settings(Config)
   print(config)  # Should show mode, thresholds, etc.
   ```

2. **Check OpenAI client availability**
   ```python
   print(f"OpenAI client: {context.get('openai_client')}")
   # Should not be None for BALANCED/THOROUGH
   ```

3. **Check evaluation is being called**
   ```python
   # Add print statement in execute_search_documents
   print("[DEBUG] Running self-reflection evaluation...")
   ```

4. **Check evaluation result**
   ```python
   print(f"[DEBUG] Eval result: {eval_result.to_dict()}")
   ```

5. **Verify USE_SELF_REFLECTION is enabled**
   ```bash
   # In .env file:
   USE_SELF_REFLECTION=true
   ```

---

## Next Steps

After completing Day 2:
- Day 3: Action Taking (Week 2) - Actually use evaluation results to trigger query refinement
- Day 4: External Search Integration (Week 3) - Connect to MCP tools based on EXTERNAL recommendation
- Day 5: Logging and Analytics - Track evaluation metrics over time

---

## Summary

By end of Day 2, you will have:
1. ✅ Working RetrievalEvaluator with 3 modes
2. ✅ Integration with agent search tool
3. ✅ Evaluation results logged for every search
4. ✅ Foundation for Week 2 action-taking

**Remember**: Week 1 is evaluation ONLY. Don't implement automatic query refinement or external search yet - that comes in Week 2 & 3.

Good luck! 🚀
