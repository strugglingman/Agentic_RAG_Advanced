# Task 1: Observability with LangSmith

**Goal**: Add production-grade observability to see what the agent is doing in real-time

**Time Estimate**: 2-3 hours
**Difficulty**: Medium
**Dependencies**: OpenAI API, LangSmith account (free tier)

---

## 📋 **Table of Contents**

1. [Overview](#overview)
2. [Why Observability?](#why-observability)
3. [Architecture Design](#architecture-design)
4. [Prerequisites](#prerequisites)
5. [Implementation Plan](#implementation-plan)
6. [Configuration](#configuration)
7. [Testing Strategy](#testing-strategy)
8. [Success Criteria](#success-criteria)
9. [Troubleshooting](#troubleshooting)

---

## 1. Overview

### **What is Observability?**

Observability = Ability to see **inside** your system while it's running.

**Without observability** (Current state):
```
User: "What is our vacation policy?"
    ↓
Agent does something...
    ↓
Response appears

❓ What did the agent do?
❓ Which tools were called?
❓ How long did it take?
❓ How many tokens used?
❓ Where did it fail?
```

**With observability** (After this task):
```
User: "What is our vacation policy?"
    ↓
[TRACE] Agent received query
[TRACE] Called search_documents tool
[TRACE]   - Retrieved 5 contexts
[TRACE]   - Self-reflection: confidence=0.85, recommendation=ANSWER
[TRACE]   - Took 234ms, used 1,250 tokens
[TRACE] Generated final answer
[TRACE]   - Took 456ms, used 320 tokens
    ↓
Response appears

✅ Full visibility into agent behavior
```

---

### **What We'll Add**

1. **Tracing**: Track every agent step (tool calls, LLM calls)
2. **Metrics**: Count tokens, measure latency, track errors
3. **Logging**: Structured logs for debugging
4. **UI**: Visual dashboard to explore traces

---

## 2. Why Observability?

### **Problem: Current State**

```python
# Current debugging (painful):
print(f"[DEBUG] Something happened")  # ← Lost in logs
print(f"[DEBUG] Token count: {???}")  # ← Don't even track this
print(f"[DEBUG] Error: {e}")          # ← Hard to find cause

# Questions you CAN'T answer:
# - Why did this query take 10 seconds?
# - Which tool call used the most tokens?
# - What percentage of queries use web_search?
# - Where is the bottleneck?
```

### **Solution: LangSmith**

```
LangSmith Dashboard:
┌────────────────────────────────────────────────────┐
│ Trace: "What is our vacation policy?"             │
│ ├─ Agent.run_stream (2.3s, 1,570 tokens)          │
│ │  ├─ search_documents (1.8s, 1,250 tokens)       │
│ │  │  ├─ retrieve() (234ms)                       │
│ │  │  ├─ evaluate() (89ms, 180 tokens)            │
│ │  │  └─ contexts: 5 found, confidence: 0.85      │
│ │  └─ LLM call (456ms, 320 tokens)                │
│ └─ Total cost: $0.0023                            │
└────────────────────────────────────────────────────┘

✅ Click any step to see inputs/outputs
✅ See token breakdown
✅ Identify slow operations
✅ Track errors with full context
```

---

### **Benefits**

| Benefit | Example |
|---------|---------|
| **Debugging** | "Why did this query fail?" → See exact error + context |
| **Performance** | "Why is this slow?" → See which step took 5 seconds |
| **Cost tracking** | "How much am I spending?" → See token usage per query |
| **Quality** | "Is self-reflection working?" → See confidence scores |
| **Product insights** | "Which tools are most used?" → Analytics dashboard |

---

## 3. Architecture Design

### **Current Architecture (No Observability)**

```
User Query
    ↓
┌──────────────────────────────────┐
│  Agent.run_stream()              │  ← Black box
│  ├─ Tool calls (invisible)       │
│  ├─ LLM calls (invisible)        │
│  └─ Return answer                │
└──────────────────────────────────┘
    ↓
Response

Logs: print() statements only
Metrics: None
Debugging: Add more print() statements
```

---

### **Target Architecture (With LangSmith)**

```
User Query
    ↓
┌──────────────────────────────────────────────────────┐
│  LangSmith Tracer (wraps everything)                 │
│  ├─ Agent.run_stream() [TRACED]                      │
│  │  ├─ search_documents() [TRACED]                   │
│  │  │  ├─ retrieve() [TRACED]                        │
│  │  │  ├─ evaluate() [TRACED]                        │
│  │  │  └─ refine_query() [TRACED] (if needed)        │
│  │  ├─ web_search() [TRACED] (if called)             │
│  │  └─ LLM.chat.completions.create() [AUTO-TRACED]   │
│  └─ All steps logged to LangSmith cloud              │
└──────────────────────────────────────────────────────┘
    ↓
Response + Full trace in LangSmith dashboard

Logs: Structured JSON to LangSmith
Metrics: Tokens, latency, cost (auto-tracked)
Debugging: View trace in UI, click on any step
```

---

### **How LangSmith Works**

```python
# 1. Decorator wraps functions
@traceable  # ← This decorator
def agent_function():
    # Your code runs normally
    # LangSmith records inputs, outputs, timing
    pass

# 2. OpenAI calls auto-traced (no changes needed)
response = openai_client.chat.completions.create(...)
# ↑ LangSmith automatically intercepts and logs this

# 3. Everything sent to LangSmith cloud
# ↓
# LangSmith Dashboard shows:
# - Function call tree
# - Inputs/outputs
# - Token counts
# - Latency
# - Errors with stack traces
```

---

## 4. Prerequisites

### **4.1 LangSmith Account** (Free Tier)

**Sign up**: https://smith.langchain.com/

**Free tier includes**:
- 5,000 traces/month
- 14-day data retention
- Full feature access

**Steps**:
1. Go to https://smith.langchain.com/
2. Sign up with email or GitHub
3. Create a project (e.g., "rag-chatbot-dev")
4. Get API key from settings

---

### **4.2 Install Dependencies**

**Add to `requirements.txt`**:
```
langsmith>=0.1.0
```

**Install**:
```bash
pip install langsmith
```

---

### **4.3 Environment Variables**

**Add to `.env`**:
```bash
# LangSmith Observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_your_api_key_here
LANGCHAIN_PROJECT=rag-chatbot-dev
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

---

## 5. Implementation Plan

### **Phase 1: Basic Tracing** (30 minutes)

**Goal**: Trace agent loop only

**Files to modify**:
1. `backend/src/services/agent_service.py`

**Changes**:
```python
# Add decorator to main agent method
from langsmith import traceable

@traceable
def run_stream(self, query, context, history):
    # Existing code (no changes needed)
    pass
```

**Test**: Run a query, check LangSmith dashboard for trace

---

### **Phase 2: Tool Tracing** (45 minutes)

**Goal**: Trace all tool executions

**Files to modify**:
1. `backend/src/services/agent_tools.py`

**Changes**:
```python
# Add decorator to tool execution functions
@traceable
def execute_search_documents(args, context):
    # Existing code
    pass

@traceable
def execute_web_search(args, context):
    # Existing code
    pass

@traceable
def execute_calculator(args, context):
    # Existing code
    pass
```

**Test**: Tool calls should appear as nested spans in trace

---

### **Phase 3: Self-Reflection Tracing** (30 minutes)

**Goal**: Trace evaluation and refinement

**Files to modify**:
1. `backend/src/services/retrieval_evaluator.py`
2. `backend/src/services/query_refiner.py`

**Changes**:
```python
# Evaluator
@traceable
def evaluate(self, criteria):
    # Existing code
    pass

# Refiner
@traceable
def refine_query(self, original_query, eval_result, context_hint):
    # Existing code
    pass
```

**Test**: See evaluation steps and refinement loops in trace

---

### **Phase 4: Custom Metadata** (30 minutes)

**Goal**: Add custom metrics (confidence scores, recommendation, etc.)

**Changes**:
```python
from langsmith import traceable

@traceable(
    metadata=lambda criteria: {
        "mode": criteria.mode.value,
        "context_count": len(criteria.contexts)
    }
)
def evaluate(self, criteria):
    result = self._evaluate_fast(criteria)

    # Log custom metrics
    langsmith.log_feedback({
        "confidence": result.confidence,
        "quality": result.quality.value,
        "recommendation": result.recommendation.value
    })

    return result
```

**Test**: Custom fields visible in LangSmith UI

---

### **Phase 5: Error Tracking** (15 minutes)

**Goal**: Automatic error capture with context

**Changes**:
```python
@traceable
def execute_search_documents(args, context):
    try:
        # Existing code
        pass
    except Exception as e:
        # LangSmith auto-captures exception
        # No extra code needed!
        raise  # Re-raise to propagate error
```

**Test**: Trigger an error, see full stack trace in LangSmith

---

## 6. Configuration

### **6.1 Environment Setup**

**Development** (`.env`):
```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_dev_key
LANGCHAIN_PROJECT=rag-chatbot-dev
```

**Production** (`.env.production`):
```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_prod_key
LANGCHAIN_PROJECT=rag-chatbot-prod
```

**Disable for Testing** (`.env.test`):
```bash
LANGCHAIN_TRACING_V2=false  # No tracing in tests
```

---

### **6.2 Sampling Configuration**

**For high-traffic production** (optional):
```python
# Only trace 10% of requests to save costs
import random
from langsmith import traceable

@traceable(enabled=lambda: random.random() < 0.1)  # 10% sampling
def run_stream(self, query, context, history):
    pass
```

---

### **6.3 Privacy Configuration**

**Redact sensitive data** (optional):
```python
from langsmith import traceable

@traceable(
    # Don't log user content to cloud
    redact_inputs=["query", "user_message"],
    redact_outputs=["answer"]
)
def run_stream(self, query, context, history):
    pass
```

---

## 7. Testing Strategy

### **7.1 Manual Testing**

**Test Case 1: Basic Query**
```
1. Start Flask: flask run
2. Send query: "What is our vacation policy?"
3. Check LangSmith dashboard
4. Verify trace shows:
   - Agent call
   - search_documents call
   - LLM call
   - Token counts
   - Latency
```

**Test Case 2: Refinement Flow**
```
1. Send poor query: "benefits"
2. Check LangSmith dashboard
3. Verify trace shows:
   - Initial search
   - Evaluation (low confidence)
   - Refinement loop
   - Improved search
   - Final answer
```

**Test Case 3: External Search**
```
1. Send external query: "current inflation rate"
2. Check LangSmith dashboard
3. Verify trace shows:
   - search_documents call
   - EXTERNAL detection
   - web_search call
   - DuckDuckGo results
```

**Test Case 4: Error Handling**
```
1. Trigger error (e.g., invalid API key)
2. Check LangSmith dashboard
3. Verify trace shows:
   - Error message
   - Stack trace
   - Full context
```

---

### **7.2 Automated Testing**

**Create test script**: `backend/tests/test_observability.py`

```python
"""
Test observability integration.
"""
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"

def test_agent_traced():
    """Verify agent calls are traced."""
    # Make agent call
    # Check that trace was created
    # Assert on trace structure
    pass

def test_tool_calls_traced():
    """Verify tool calls are traced."""
    pass

def test_evaluation_traced():
    """Verify self-reflection is traced."""
    pass
```

---

## 8. Success Criteria

### **Must Have** ✅

- [ ] Agent calls visible in LangSmith dashboard
- [ ] Tool calls shown as nested spans
- [ ] Token counts tracked per call
- [ ] Latency measured for each step
- [ ] Errors captured with full context
- [ ] Can replay any conversation from dashboard

### **Should Have** ⭐

- [ ] Self-reflection metrics visible (confidence, quality)
- [ ] Refinement loops clearly shown
- [ ] Web search calls tracked
- [ ] Custom metadata (recommendation, context_count)

### **Nice to Have** 🎯

- [ ] Cost tracking per query
- [ ] Performance analytics (p50, p95, p99)
- [ ] User feedback integration
- [ ] Alerts for errors/slow queries

---

## 9. Troubleshooting

### **Issue 1: Traces not appearing in dashboard**

**Check**:
1. `LANGCHAIN_TRACING_V2=true` in `.env`
2. API key is valid
3. Internet connection working
4. Project name matches dashboard

**Debug**:
```python
import os
print(f"Tracing enabled: {os.getenv('LANGCHAIN_TRACING_V2')}")
print(f"API key set: {bool(os.getenv('LANGCHAIN_API_KEY'))}")
print(f"Project: {os.getenv('LANGCHAIN_PROJECT')}")
```

---

### **Issue 2: Only some calls traced**

**Cause**: Forgot `@traceable` decorator on some functions

**Fix**: Add decorator to all functions you want traced

---

### **Issue 3: Sensitive data in traces**

**Cause**: Default behavior logs all inputs/outputs

**Fix**: Use `redact_inputs` and `redact_outputs`:
```python
@traceable(redact_inputs=["password", "api_key"])
def my_function(password, data):
    pass
```

---

### **Issue 4: High costs (too many traces)**

**Cause**: Tracing every request in high-traffic production

**Fix**: Enable sampling:
```python
@traceable(enabled=lambda: random.random() < 0.1)  # 10%
```

---

### **Issue 5: Slow performance**

**Cause**: LangSmith adds ~5-10ms overhead per trace

**Fix**: This is normal and acceptable. For ultra-low-latency needs, use async tracing or sampling.

---

## 10. File Checklist

### **Files to Modify**:

```
backend/
├── .env                                    # Add LangSmith config
├── requirements.txt                        # Add langsmith
├── src/
│   ├── services/
│   │   ├── agent_service.py               # Add @traceable to run_stream
│   │   ├── agent_tools.py                 # Add @traceable to all execute_* functions
│   │   ├── retrieval_evaluator.py         # Add @traceable to evaluate
│   │   └── query_refiner.py               # Add @traceable to refine_query
│   └── routes/
│       └── chat.py                         # (Optional) Add @traceable to chat endpoint
└── tests/
    └── test_observability.py               # Create new test file
```

### **Files to Create**:

```
backend/
└── docs/
    └── OBSERVABILITY_SETUP.md              # Setup instructions for team
```

---

## 11. Cost Analysis

### **LangSmith Pricing**

**Free Tier**:
- 5,000 traces/month
- 14-day retention
- Perfect for development + small production

**Developer Tier** ($39/month):
- 100,000 traces/month
- 400-day retention
- Team collaboration

**Cost per trace**: ~$0.00039 (Developer tier)

### **ROI Analysis**

**Without observability**:
- Bug takes 2 hours to debug (can't see what happened)
- Cost: $100 (engineer time)

**With observability**:
- Bug takes 10 minutes to debug (full trace visible)
- Cost: $0.00039 (trace) + $8 (engineer time)
- **Savings**: $92 per bug

**Pays for itself after 1 bug!** 🎉

---

## 12. Next Steps After Implementation

### **Week 1**: Basic usage
- Explore dashboard
- Understand trace structure
- Fix any errors found

### **Week 2**: Advanced features
- Add custom metadata
- Set up alerts
- Create dashboards

### **Week 3**: Optimization
- Analyze slow queries
- Identify bottlenecks
- Optimize token usage

### **Week 4**: Production readiness
- Enable sampling (10-20%)
- Set up error alerts
- Create runbooks

---

## 13. References

**LangSmith Documentation**:
- Main docs: https://docs.smith.langchain.com/
- Python SDK: https://docs.smith.langchain.com/tracing/sdk/python
- Tracing guide: https://docs.smith.langchain.com/tracing

**Tutorials**:
- Quick start: https://docs.smith.langchain.com/tracing/quick_start
- OpenAI integration: https://docs.smith.langchain.com/tracing/integrations/openai

**Examples**:
- Sample traces: https://smith.langchain.com/public/
- Best practices: https://docs.smith.langchain.com/tracing/best_practices

---

## 14. Summary

### **What You'll Get**:

**Before** (Current):
- ❌ No visibility into agent behavior
- ❌ Hard to debug issues
- ❌ No token tracking
- ❌ No performance metrics
- ❌ Print-based logging

**After** (With LangSmith):
- ✅ Full trace of every agent step
- ✅ Click to debug any issue
- ✅ Token counts per call
- ✅ Latency metrics
- ✅ Beautiful visual dashboard
- ✅ Error tracking with context
- ✅ Cost analysis
- ✅ Performance optimization insights

**Time to implement**: 2-3 hours
**Value added**: Production-grade observability
**Resume impact**: Shows production awareness

---

**Ready to implement? Review this guide, then we'll proceed with code changes!** 🚀
