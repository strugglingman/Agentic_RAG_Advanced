# Week 3: External Search - Web Fallback System

**Status**: 📋 Planning Phase
**Estimated Time**: 2-3 days
**Prerequisite**: Week 2 Complete ✅

---

## 🎯 Goals

Implement **external web search fallback** when internal documents don't contain the requested information.

**Week 1 Recap**: System evaluates retrieval quality and logs recommendations
**Week 2 Recap**: System takes action on REFINE/CLARIFY recommendations
**Week 3 Goal**: System **falls back to web search** when EXTERNAL recommendation is given

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
│ search_documents     │ ← Retrieves from internal docs
└──────┬───────────────┘
       │
       ▼
┌──────────────────────────────┐
│ RetrievalEvaluator.evaluate()│ ← Evaluates quality
└──────┬───────────────────────┘
       │
       ▼
  ┌────────────────────────────────────┐
  │ Decision Logic:                    │
  │                                    │
  │ • ANSWER → return contexts         │ ✅ Week 1
  │ • REFINE → reformulate + retry     │ ✅ Week 2
  │ • CLARIFY → ask user for details   │ ✅ Week 2
  │ • EXTERNAL → search web fallback   │ ⬅ Week 3: Implement this
  └────────────────────────────────────┘
```

---

## 🎬 User Experience Flow

### Scenario: Information Not in Documents → Web Search

```
User: "What is the current inflation rate?"
  ↓
Agent searches internal documents → no relevant contexts
  ↓
Evaluator: Quality=POOR, Recommendation=EXTERNAL
  Reason: "Query appears to be about real-time/external data not in documents"
  ↓
Agent detects EXTERNAL recommendation
  ↓
Agent calls web_search tool → fetches current information
  ↓
Agent returns answer with web sources ✅
```

---

## 🔑 Key Features to Implement

### 1. **Web Search Tool** (Priority 1)

Add a new tool that the agent can use to search the web.

**Options**:
- **Tavily API** - Built for AI agents, returns structured results
- **SerpAPI** - Google search results
- **Brave Search API** - Privacy-focused, good free tier
- **DuckDuckGo** - Free, no API key needed (via `duckduckgo-search` package)

**Recommendation**: Start with **DuckDuckGo** (free, no API key) or **Tavily** (best for AI agents)

### 2. **EXTERNAL Recommendation Detection** (Priority 2)

Update `agent_tools.py` to detect when evaluator recommends EXTERNAL search.

```python
# When eval_result.recommendation == RecommendationAction.EXTERNAL:
#   - Don't try refinement (won't help - info not in docs)
#   - Return a signal to agent to use web_search tool
```

### 3. **Hybrid Response** (Priority 3)

Combine internal document results with web search results when appropriate.

---

## 📁 Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/services/web_search.py` | CREATE | Web search service |
| `src/services/agent_tools.py` | MODIFY | Add web_search tool schema + execution |
| `src/models/evaluation.py` | CHECK | Ensure EXTERNAL recommendation exists |
| `src/config/settings.py` | MODIFY | Add web search config |
| `.env` | MODIFY | Add web search API keys (if needed) |

---

## 🏗️ Implementation Plan

### Day 5: Web Search Service
- Create `web_search.py` service
- Implement search provider (DuckDuckGo or Tavily)
- Add result formatting
- Test standalone

### Day 6: Agent Integration
- Add `web_search` tool schema to agent_tools.py
- Implement `execute_web_search()` function
- Handle EXTERNAL recommendation in search_documents
- Test end-to-end flow

### Day 7 (Optional): Advanced Features
- Hybrid responses (combine docs + web)
- Source attribution
- Caching for repeated queries
- Rate limiting

---

## ⚠️ Important Considerations

### 1. When to Use Web Search

Web search should be triggered when:
- Query is about **real-time data** (prices, weather, news)
- Query is about **external entities** (other companies, public figures)
- Internal documents have **no relevant results**
- Evaluator explicitly recommends EXTERNAL

### 2. When NOT to Use Web Search

Avoid web search for:
- **Confidential company data** (policies, financials)
- **Internal processes** (how to submit expense report)
- Queries where internal docs should have the answer

### 3. Security Considerations

- Don't send confidential query content to external search APIs
- Clearly mark web results as "external source"
- Consider user permission before searching web

---

## 🔧 Configuration Options

```python
# settings.py
WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "false").lower() in {"1", "true", "yes"}
WEB_SEARCH_PROVIDER = os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo")  # duckduckgo, tavily, serp
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")  # If using Tavily
```

---

## 📊 Decision Flow After Week 3

```
Evaluation Result:
├── confidence ≥ 0.70 (ANSWER)         → Return contexts directly
├── 0.50 ≤ confidence < 0.70 (REFINE)  → Try refinement loop
│   ├── Refinement succeeds            → Return improved contexts
│   └── Max attempts reached           → Clarification message
├── confidence < 0.50 (CLARIFY)        → Clarification message
└── EXTERNAL recommendation            → Web search fallback (NEW)
```

---

## 🚀 Quick Start

After completing this week, the agent will:

1. Search internal documents first
2. Evaluate retrieval quality
3. If EXTERNAL recommended → automatically search the web
4. Return combined/web results with source attribution

---

## 📚 References

- [Tavily API Docs](https://docs.tavily.com/)
- [DuckDuckGo Search Package](https://pypi.org/project/duckduckgo-search/)
- Week 2 Overview: [WEEK2_OVERVIEW.md](./WEEK2_OVERVIEW.md)
- Evaluation Models: [../src/models/evaluation.py](../src/models/evaluation.py)
