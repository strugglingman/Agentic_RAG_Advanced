# Week 3 - Day 6: Agent Integration for Web Search

**Status**: 📋 TODO
**Focus**: Integrate web search tool into agent, handle EXTERNAL recommendation
**Time**: 3-4 hours
**Prerequisites**: Day 5 Complete ✅

---

## 📋 Overview

Day 6 integrates the web search service into the agent:
1. **Web search tool schema**: Add tool definition for agent
2. **EXTERNAL detection**: Handle EXTERNAL recommendation from evaluator
3. **Tool execution**: Execute web search when agent calls the tool
4. **End-to-end testing**: Verify complete flow

---

## 🎯 What to Build

### Current State (After Day 5):

```python
# WebSearchService exists but not connected to agent
# Agent has no web_search tool
# EXTERNAL recommendation is logged but not acted upon
```

### Target State (After Day 6):

```python
# Agent can use web_search tool
tools = [
    {"type": "function", "function": search_documents_schema},
    {"type": "function", "function": web_search_schema},  # NEW
]

# EXTERNAL recommendation triggers web search suggestion
if eval_result.recommendation == RecommendationAction.EXTERNAL:
    # Return signal to agent to use web_search tool
```

---

## 🏗️ Part 1: Add Web Search Tool Schema

### Step 1.1: Define tool schema

**File**: `backend/src/services/agent_tools.py`

**Add after search_documents_schema (around line 60)**:

```python
# =============================================================================
# WEB SEARCH TOOL SCHEMA (Week 3)
# =============================================================================

web_search_schema = {
    "name": "web_search",
    "description": """Search the web for external information not found in uploaded documents.

Use this tool when:
- Query is about real-time data (prices, weather, news)
- Query is about external entities not in documents
- Internal document search returned no relevant results
- Evaluator recommended EXTERNAL search

Do NOT use for:
- Confidential company data
- Internal processes/policies
- Information that should be in uploaded documents""",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query for web search"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 5)",
                "default": 5
            }
        },
        "required": ["query"]
    }
}
```

### Step 1.2: Add import

**File**: `backend/src/services/agent_tools.py`

**Add to imports (around line 22)**:
```python
from src.services.web_search import WebSearchService, WebSearchResult
```

---

## 🏗️ Part 2: Implement Web Search Execution

### Step 2.1: Create execute_web_search function

**File**: `backend/src/services/agent_tools.py`

**Add after execute_search_documents function**:

```python
def execute_web_search(
    query: str,
    max_results: int = 5,
    context: Dict[str, Any] = None,
) -> str:
    """
    Execute web search tool.

    Args:
        query: Search query
        max_results: Maximum results to return
        context: Agent context (optional)

    Returns:
        Formatted search results string

    TODO: Implement web search execution
    Steps:
    1. Check if Config.WEB_SEARCH_ENABLED:
       - If False, return "Web search is disabled. Enable with WEB_SEARCH_ENABLED=true"
    2. Print: f"[WEB_SEARCH] Searching web for: {query}"
    3. Try:
       a. Create service: service = WebSearchService()
       b. Search: results = service.search(query, max_results=max_results)
       c. Format: formatted = service.format_for_agent(results, query)
       d. Print: f"[WEB_SEARCH] Found {len(results)} results"
       e. Return formatted
    4. Except Exception as e:
       - Print: f"[WEB_SEARCH] Error: {e}"
       - Return f"Web search failed: {str(e)}"
    """
    pass
```

---

## 🏗️ Part 3: Handle EXTERNAL Recommendation

### Step 3.1: Update search_documents to signal EXTERNAL

**File**: `backend/src/services/agent_tools.py`

**Location**: After evaluation in execute_search_documents (after refinement/clarification logic)

**Add EXTERNAL handling**:

```python
            # =============================================================================
            # EXTERNAL RECOMMENDATION HANDLING (Week 3)
            # =============================================================================
            # If evaluator recommends EXTERNAL search, signal agent to use web_search tool
            if eval_result.recommendation == RecommendationAction.EXTERNAL:
                print(f"[SELF-REFLECTION] EXTERNAL recommended - suggesting web search")
                external_msg = (
                    f"[EXTERNAL SEARCH SUGGESTED]\n"
                    f"The query \"{query}\" appears to require external information "
                    f"not found in the uploaded documents.\n\n"
                    f"Reason: {eval_result.reasoning}\n\n"
                    f"Please use the web_search tool to find this information.\n\n"
                    f"---\n\n"
                )
                # Prepend to result (or return alone if no contexts)
                if contexts:
                    result = external_msg + result
                else:
                    result = external_msg + "No relevant documents found."
```

### Step 3.2: Update RecommendationAction enum (if needed)

**File**: `backend/src/models/evaluation.py`

**Verify EXTERNAL exists in enum**:

```python
class RecommendationAction(str, Enum):
    """Recommended action based on evaluation."""
    ANSWER = "answer"      # Proceed with answer
    REFINE = "refine"      # Try query refinement
    CLARIFY = "clarify"    # Ask user for clarification
    EXTERNAL = "external"  # Search external sources
```

---

## 🏗️ Part 4: Update Tool Registry

### Step 4.1: Add web_search to available tools

**File**: `backend/src/services/agent_tools.py`

**Update get_tools function**:

```python
def get_tools() -> List[Dict]:
    """
    Get list of available tools for agent.

    Returns:
        List of tool schemas

    TODO: Update to include web_search
    Steps:
    1. Start with: tools = [{"type": "function", "function": search_documents_schema}]
    2. If Config.WEB_SEARCH_ENABLED:
       - Append: tools.append({"type": "function", "function": web_search_schema})
    3. Return tools
    """
    tools = [
        {"type": "function", "function": search_documents_schema},
    ]

    if Config.WEB_SEARCH_ENABLED:
        tools.append({"type": "function", "function": web_search_schema})

    return tools
```

### Step 4.2: Update execute_tool function

**File**: `backend/src/services/agent_tools.py`

**Add web_search handling to execute_tool**:

```python
def execute_tool(
    tool_name: str,
    tool_args: Dict[str, Any],
    context: Dict[str, Any],
) -> str:
    """
    Execute a tool by name.

    TODO: Add web_search handling
    """
    if tool_name == "search_documents":
        return execute_search_documents(
            query=tool_args.get("query", ""),
            collection_name=tool_args.get("collection_name"),
            context=context,
        )
    elif tool_name == "web_search":  # NEW
        return execute_web_search(
            query=tool_args.get("query", ""),
            max_results=tool_args.get("max_results", 5),
            context=context,
        )
    else:
        return f"Unknown tool: {tool_name}"
```

---

## 🏗️ Part 5: Update Evaluator for EXTERNAL

### Step 5.1: Ensure evaluator can recommend EXTERNAL

**File**: `backend/src/services/retrieval_evaluator.py`

**Add EXTERNAL detection logic in evaluate method**:

```python
# In _determine_recommendation method or evaluate logic:

# Check for EXTERNAL indicators
external_indicators = [
    "current",
    "latest",
    "today",
    "real-time",
    "live",
    "weather",
    "stock price",
    "news",
    "trending",
]

query_lower = query.lower()
is_external_query = any(indicator in query_lower for indicator in external_indicators)

# If no relevant contexts AND query seems external
if confidence < 0.3 and is_external_query:
    return RecommendationAction.EXTERNAL
```

---

## 🧪 Part 6: Testing

### Test 1: Tool schema verification

```python
from src.services.agent_tools import get_tools

tools = get_tools()
print(f"Available tools: {len(tools)}")
for t in tools:
    print(f"  - {t['function']['name']}")
```

### Test 2: Web search execution

```python
from src.services.agent_tools import execute_web_search

result = execute_web_search("Python programming language", max_results=3)
print(result)
```

### Test 3: End-to-end flow

```
User: "What is the current inflation rate?"
  ↓
Agent calls search_documents
  ↓
Evaluator: confidence=0.15, recommendation=EXTERNAL
  ↓
Agent receives: "[EXTERNAL SEARCH SUGGESTED]..."
  ↓
Agent calls web_search("current inflation rate")
  ↓
Agent returns answer with web sources
```

### Test 4: Manual integration test

```bash
cd backend
python tests/test_web_search_integration.py
```

---

## ✅ Day 6 Checklist

### Part 1: Tool Schema
- [ ] Add `web_search_schema` to agent_tools.py
- [ ] Add WebSearchService import

### Part 2: Execution
- [ ] Implement `execute_web_search()` function
- [ ] Handle errors gracefully

### Part 3: EXTERNAL Handling
- [ ] Add EXTERNAL detection in search_documents
- [ ] Signal agent to use web_search tool
- [ ] Verify RecommendationAction.EXTERNAL exists

### Part 4: Tool Registry
- [ ] Update `get_tools()` to include web_search
- [ ] Update `execute_tool()` to handle web_search

### Part 5: Evaluator
- [ ] Add EXTERNAL detection logic
- [ ] Test external query detection

### Part 6: Testing
- [ ] Verify tool schemas load correctly
- [ ] Test web search execution
- [ ] Test end-to-end flow
- [ ] Test with real queries

---

## 🎯 Success Criteria

Day 6 is complete when:

- [ ] Agent has access to `web_search` tool
- [ ] EXTERNAL recommendation triggers suggestion
- [ ] Web search returns formatted results
- [ ] End-to-end flow works correctly
- [ ] All integration tests pass

---

## 📊 Expected Log Output

### Scenario: External query flow

```
[SEARCH] Query: "What is the current inflation rate?"
[SEARCH] Searching collection: documents
[SEARCH] Retrieved 3 contexts

[SELF-REFLECTION] Evaluating retrieval quality...
[SELF-REFLECTION] Quality: poor, Confidence: 0.18
[SELF-REFLECTION] Recommendation: EXTERNAL
[SELF-REFLECTION] EXTERNAL recommended - suggesting web search

--- Agent decides to use web_search ---

[WEB_SEARCH] Searching web for: current inflation rate
[WEB_SEARCH] Found 5 results

Web search results for: "current inflation rate"

1. US Inflation Rate - Bureau of Labor Statistics
   Source: bls.gov
   URL: https://www.bls.gov/cpi/
   The Consumer Price Index for All Urban Consumers...

[Note: These results are from external web sources]
```

---

## ⚠️ Security Considerations

### 1. Query sanitization

Don't send sensitive information to external search:

```python
def _sanitize_query(self, query: str) -> str:
    """Remove potentially sensitive information from query."""
    # Remove email addresses
    # Remove phone numbers
    # Remove company-specific terms
    return sanitized_query
```

### 2. Result attribution

Always clearly mark external sources:

```python
"[Note: These results are from external web sources]"
```

### 3. User awareness

Consider adding user confirmation for external searches (optional):

```python
# In future: Ask user permission before searching web
```

---

## 📚 References

- Day 5 Implementation: [WEEK3_DAY5_IMPLEMENTATION.md](./WEEK3_DAY5_IMPLEMENTATION.md)
- Web Search Service: [../src/services/web_search.py](../src/services/web_search.py)
- Agent Tools: [../src/services/agent_tools.py](../src/services/agent_tools.py)
- Evaluation Models: [../src/models/evaluation.py](../src/models/evaluation.py)

---

## 🚀 After Day 6: Week 3 Complete!

Once Day 6 is done, Week 3 features are complete:

| Feature | Day | Status |
|---------|-----|--------|
| Web Search Service | Day 5 | 📋 TODO |
| Agent Integration | Day 6 | 📋 TODO |
| EXTERNAL Handling | Day 6 | 📋 TODO |
| End-to-end Testing | Day 6 | 📋 TODO |

### Optional Day 7 Enhancements:

- [ ] Hybrid responses (combine docs + web)
- [ ] Caching for repeated queries
- [ ] Rate limiting
- [ ] Additional search providers (Tavily, SerpAPI)
