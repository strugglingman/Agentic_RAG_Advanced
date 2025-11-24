# Week 3 Implementation Checklist

**Status**: 📋 TODO
**Estimated Time**: 2-3 days (6-8 hours total)
**Prerequisites**: Week 2 Complete ✅

---

## Quick Reference

| Day | Focus | Time | Status |
|-----|-------|------|--------|
| **Day 5** | Web Search Service | 3-4h | 📋 TODO |
| **Day 6** | Agent Integration | 3-4h | 📋 TODO |
| **Day 7** | Advanced Features (Optional) | 2-3h | 📋 Optional |

---

## Day 5: Web Search Service 📋 TODO

### Part 1: Dependencies

- [ ] Install `duckduckgo-search` package
- [ ] Update `requirements.txt`

### Part 2: Configuration

- [ ] Add to `settings.py`:
  - [ ] `WEB_SEARCH_ENABLED`
  - [ ] `WEB_SEARCH_PROVIDER`
  - [ ] `WEB_SEARCH_MAX_RESULTS`
  - [ ] `TAVILY_API_KEY` (optional)
- [ ] Add settings to `.env`

### Part 3: WebSearchService

- [ ] Create `src/services/web_search.py`
- [ ] Implement `WebSearchResult` dataclass
- [ ] Implement `WebSearchService` class:
  - [ ] `__init__(provider, max_results, tavily_api_key)`
  - [ ] `search(query, max_results)`
  - [ ] `_validate_query(query)`
  - [ ] `_search_duckduckgo(query, max_results)`
  - [ ] `_extract_domain(url)`
  - [ ] `format_for_agent(results, query)`
- [ ] (Optional) Implement `_search_tavily()`

### Part 4: Testing

- [ ] Add test block to `web_search.py`
- [ ] Run standalone tests: `python -m src.services.web_search`
- [ ] Verify DuckDuckGo returns results
- [ ] Test with various queries

---

## Day 6: Agent Integration 📋 TODO

### Part 1: Tool Schema

- [ ] Add `web_search_schema` to `agent_tools.py`
- [ ] Import `WebSearchService`

### Part 2: Execution Function

- [ ] Implement `execute_web_search()` function
- [ ] Handle disabled state gracefully
- [ ] Handle errors gracefully

### Part 3: EXTERNAL Recommendation

- [ ] Add EXTERNAL handling in `execute_search_documents()`
- [ ] Signal agent to use web_search tool
- [ ] Verify `RecommendationAction.EXTERNAL` exists in enum

### Part 4: Tool Registry

- [ ] Update `get_tools()` to include web_search (when enabled)
- [ ] Update `execute_tool()` to handle web_search calls

### Part 5: Evaluator Updates

- [ ] Add EXTERNAL detection logic to evaluator
- [ ] Define external query indicators
- [ ] Test external query detection

### Part 6: End-to-End Testing

- [ ] Test tool schema loading
- [ ] Test web search execution
- [ ] Test EXTERNAL recommendation flow
- [ ] Test complete agent interaction

---

## Day 7: Advanced Features (Optional) 📋

### Hybrid Responses

- [ ] Combine document results with web results
- [ ] Prioritize internal docs over web results
- [ ] Implement result merging logic

### Caching

- [ ] Add cache for repeated queries
- [ ] Set cache expiration (e.g., 1 hour)
- [ ] Clear cache on service restart

### Rate Limiting

- [ ] Add delay between searches
- [ ] Implement rate limit counter
- [ ] Handle rate limit errors gracefully

### Additional Providers

- [ ] Implement Tavily provider
- [ ] Implement SerpAPI provider
- [ ] Add provider selection logic

---

## Files Summary

### To Create:

```
src/services/web_search.py              📋 Day 5 - Web search service
```

### To Modify:

```
src/config/settings.py                  📋 Day 5 - Web search config
.env                                    📋 Day 5 - Web search settings
requirements.txt                        📋 Day 5 - Add duckduckgo-search
src/services/agent_tools.py             📋 Day 6 - Add web_search tool
src/services/retrieval_evaluator.py     📋 Day 6 - EXTERNAL detection
src/models/evaluation.py                📋 Day 6 - Verify EXTERNAL enum
```

---

## Configuration Reference

### settings.py additions:

```python
# WEB SEARCH CONFIGURATION (Week 3)
WEB_SEARCH_ENABLED: bool = os.getenv("WEB_SEARCH_ENABLED", "false").lower() in {"1", "true", "yes"}
WEB_SEARCH_PROVIDER: str = os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo")
WEB_SEARCH_MAX_RESULTS: int = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
```

### .env additions:

```bash
# WEB SEARCH CONFIGURATION (Week 3)
WEB_SEARCH_ENABLED=true
WEB_SEARCH_PROVIDER=duckduckgo
WEB_SEARCH_MAX_RESULTS=5
# TAVILY_API_KEY=your_key_here
```

---

## Testing Commands

### Run WebSearchService tests:

```bash
cd backend
python -m src.services.web_search
```

### Run integration test:

```bash
cd backend
python tests/test_web_search_integration.py
```

### Quick verification:

```python
from src.services.web_search import WebSearchService

service = WebSearchService()
results = service.search("Python programming", max_results=3)
print(f"Found {len(results)} results")
```

---

## Success Criteria

### Day 5:

- [ ] WebSearchService initializes without errors
- [ ] DuckDuckGo search returns results
- [ ] Results are properly formatted
- [ ] All standalone tests pass

### Day 6:

- [ ] Agent has access to web_search tool
- [ ] EXTERNAL recommendation triggers suggestion
- [ ] Web search executes successfully
- [ ] End-to-end flow works

### Week 3 Complete:

- [ ] Internal docs searched first
- [ ] EXTERNAL triggers web fallback
- [ ] Results properly attributed
- [ ] All scenarios tested

---

## Decision Flow After Week 3

```
Evaluation Result:
├── confidence ≥ 0.70 (ANSWER)         → Return contexts directly       ✅ Week 1
├── 0.50 ≤ confidence < 0.70 (REFINE)  → Try refinement loop            ✅ Week 2
│   ├── Refinement succeeds            → Return improved contexts
│   └── Max attempts reached           → Clarification message
├── confidence < 0.50 (CLARIFY)        → Clarification message          ✅ Week 2
└── EXTERNAL recommendation            → Web search fallback            ⬅ Week 3
```

---

## Common Issues & Solutions

### Issue: DuckDuckGo not returning results

**Check**:
1. Internet connection
2. Query not too specific
3. Add delay between requests (rate limiting)

### Issue: Web search tool not appearing

**Check**:
1. `WEB_SEARCH_ENABLED=true` in .env
2. Config loaded correctly
3. `get_tools()` includes web_search

### Issue: EXTERNAL never recommended

**Check**:
1. Evaluator has EXTERNAL detection logic
2. Query contains external indicators
3. Confidence is low enough

### Issue: Import errors

**Check**:
1. `duckduckgo-search` installed
2. Correct import path
3. Virtual environment activated

---

## References

- **Overview**: [WEEK3_OVERVIEW.md](./WEEK3_OVERVIEW.md)
- **Day 5 Guide**: [WEEK3_DAY5_IMPLEMENTATION.md](./WEEK3_DAY5_IMPLEMENTATION.md)
- **Day 6 Guide**: [WEEK3_DAY6_IMPLEMENTATION.md](./WEEK3_DAY6_IMPLEMENTATION.md)
- **Week 2 Complete**: [WEEK2_CHECKLIST.md](./WEEK2_CHECKLIST.md) ✅

---

## Progress Tracking

| Week | Feature | Status |
|------|---------|--------|
| Week 1 | Self-Reflection Evaluation | ✅ Complete |
| Week 2 | Query Refinement + Clarification | ✅ Complete |
| Week 3 | External Web Search | 📋 TODO |
| Week 4 | (Future) Advanced Features | 📋 Planning |
