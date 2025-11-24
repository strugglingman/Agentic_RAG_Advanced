# Week 3 - Day 5: Web Search Service Implementation

**Status**: 📋 TODO
**Focus**: Create web search service with DuckDuckGo provider
**Time**: 3-4 hours
**Prerequisites**: Week 2 Complete ✅

---

## 📋 Overview

Day 5 implements the core web search service:
1. **Web search service**: Create `web_search.py` with provider abstraction
2. **DuckDuckGo provider**: Free, no API key required
3. **Result formatting**: Structure results for agent consumption
4. **Standalone testing**: Verify service works independently

---

## 🎯 What to Build

### Target Architecture:

```
┌─────────────────────────────────────────┐
│           WebSearchService              │
├─────────────────────────────────────────┤
│ - search(query, max_results)            │
│ - _format_results(raw_results)          │
│ - _validate_query(query)                │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│         Search Providers                │
├─────────────────────────────────────────┤
│ • DuckDuckGo (default, free)            │
│ • Tavily (optional, API key)            │
│ • SerpAPI (optional, API key)           │
└─────────────────────────────────────────┘
```

---

## 🏗️ Part 1: Install Dependencies

### Step 1.1: Add duckduckgo-search package

```bash
cd backend
pip install duckduckgo-search
```

### Step 1.2: Update requirements.txt

**File**: `backend/requirements.txt`

**Add**:
```
duckduckgo-search>=6.0.0
```

---

## 🏗️ Part 2: Add Configuration

### Step 2.1: Update settings.py

**File**: `backend/src/config/settings.py`

**Add after REFLECTION settings**:
```python
    # =============================================================================
    # WEB SEARCH CONFIGURATION (Week 3)
    # =============================================================================
    WEB_SEARCH_ENABLED: bool = os.getenv("WEB_SEARCH_ENABLED", "false").lower() in {"1", "true", "yes"}
    WEB_SEARCH_PROVIDER: str = os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo")  # duckduckgo, tavily
    WEB_SEARCH_MAX_RESULTS: int = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")  # Optional: for Tavily provider
```

### Step 2.2: Update .env

**File**: `backend/.env`

**Add**:
```bash
# =============================================================================
# WEB SEARCH CONFIGURATION (Week 3)
# =============================================================================
WEB_SEARCH_ENABLED=true
WEB_SEARCH_PROVIDER=duckduckgo
WEB_SEARCH_MAX_RESULTS=5
# TAVILY_API_KEY=your_key_here  # Optional: uncomment if using Tavily
```

---

## 🏗️ Part 3: Create Web Search Service

### Step 3.1: Create web_search.py

**File**: `backend/src/services/web_search.py`

```python
"""
Web search service for external information retrieval.

Provides fallback search when internal documents don't contain
the requested information (EXTERNAL recommendation).

Week 3 - Day 5: Web Search Service
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from src.config.settings import Config


@dataclass
class WebSearchResult:
    """
    Represents a single web search result.

    Attributes:
        title: Page title
        url: Page URL
        snippet: Text snippet/description
        source: Source domain
    """
    title: str
    url: str
    snippet: str
    source: str


class WebSearchService:
    """
    Web search service with provider abstraction.

    Supports multiple search providers:
    - DuckDuckGo (default, free, no API key)
    - Tavily (optional, requires API key)

    Example Usage:
        service = WebSearchService()
        results = service.search("current inflation rate", max_results=5)
        for result in results:
            print(f"{result.title}: {result.url}")
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        max_results: int = 5,
        tavily_api_key: Optional[str] = None,
    ):
        """
        Initialize web search service.

        Args:
            provider: Search provider ("duckduckgo" or "tavily")
            max_results: Default maximum results to return
            tavily_api_key: API key for Tavily (if using)

        TODO: Implement initialization
        Steps:
        1. Set self.provider = provider or Config.WEB_SEARCH_PROVIDER
        2. Set self.max_results = max_results
        3. Set self.tavily_api_key = tavily_api_key or Config.TAVILY_API_KEY
        4. Validate provider is supported: ["duckduckgo", "tavily"]
        5. If provider is "tavily" and no API key, raise ValueError
        """
        pass

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> List[WebSearchResult]:
        """
        Perform web search.

        Args:
            query: Search query string
            max_results: Maximum results to return (uses default if None)

        Returns:
            List of WebSearchResult objects

        Raises:
            ValueError: If query is empty or invalid
            RuntimeError: If search fails

        TODO: Implement search
        Steps:
        1. Validate query using _validate_query()
        2. Set num_results = max_results or self.max_results
        3. If self.provider == "duckduckgo":
           - Call _search_duckduckgo(query, num_results)
        4. Elif self.provider == "tavily":
           - Call _search_tavily(query, num_results)
        5. Else:
           - Raise ValueError(f"Unsupported provider: {self.provider}")
        6. Return results
        """
        pass

    def _validate_query(self, query: str) -> str:
        """
        Validate and clean search query.

        Args:
            query: Raw query string

        Returns:
            Cleaned query string

        Raises:
            ValueError: If query is empty or too short

        TODO: Implement validation
        Steps:
        1. Strip whitespace: query = query.strip()
        2. If len(query) < 3:
           - Raise ValueError("Query too short (minimum 3 characters)")
        3. If len(query) > 500:
           - Truncate: query = query[:500]
        4. Return query
        """
        pass

    def _search_duckduckgo(
        self,
        query: str,
        max_results: int,
    ) -> List[WebSearchResult]:
        """
        Search using DuckDuckGo.

        Args:
            query: Search query
            max_results: Maximum results

        Returns:
            List of WebSearchResult

        TODO: Implement DuckDuckGo search
        Steps:
        1. Import: from duckduckgo_search import DDGS
        2. Try:
           a. Create client: ddgs = DDGS()
           b. Call: raw_results = ddgs.text(query, max_results=max_results)
           c. Format results:
              results = []
              for r in raw_results:
                  results.append(WebSearchResult(
                      title=r.get("title", ""),
                      url=r.get("href", ""),
                      snippet=r.get("body", ""),
                      source=self._extract_domain(r.get("href", "")),
                  ))
           d. Return results
        3. Except Exception as e:
           - Print: f"[WEB_SEARCH] DuckDuckGo error: {e}"
           - Return empty list []
        """
        pass

    def _search_tavily(
        self,
        query: str,
        max_results: int,
    ) -> List[WebSearchResult]:
        """
        Search using Tavily API.

        Args:
            query: Search query
            max_results: Maximum results

        Returns:
            List of WebSearchResult

        TODO: Implement Tavily search (optional)
        Steps:
        1. This is optional - implement if you have Tavily API key
        2. Import: from tavily import TavilyClient
        3. Create client: client = TavilyClient(api_key=self.tavily_api_key)
        4. Call: response = client.search(query, max_results=max_results)
        5. Format results similar to DuckDuckGo
        6. Return results

        For now, can just raise NotImplementedError or return empty list
        """
        pass

    def _extract_domain(self, url: str) -> str:
        """
        Extract domain from URL.

        Args:
            url: Full URL string

        Returns:
            Domain name (e.g., "example.com")

        TODO: Implement domain extraction
        Steps:
        1. Try:
           a. from urllib.parse import urlparse
           b. parsed = urlparse(url)
           c. domain = parsed.netloc
           d. Remove "www." prefix if present
           e. Return domain
        2. Except:
           - Return "unknown"
        """
        pass

    def format_for_agent(
        self,
        results: List[WebSearchResult],
        query: str,
    ) -> str:
        """
        Format search results for agent consumption.

        Args:
            results: List of search results
            query: Original query

        Returns:
            Formatted string for agent context

        TODO: Implement formatting
        Steps:
        1. If no results:
           - Return f"No web results found for: {query}"
        2. Build formatted string:
           output = f"Web search results for: \"{query}\"\n\n"
           for i, r in enumerate(results, 1):
               output += f"{i}. {r.title}\n"
               output += f"   Source: {r.source}\n"
               output += f"   URL: {r.url}\n"
               output += f"   {r.snippet}\n\n"
        3. Add disclaimer:
           output += "[Note: These results are from external web sources]"
        4. Return output
        """
        pass


# =============================================================================
# TESTING (run with: python -m src.services.web_search)
# =============================================================================

if __name__ == "__main__":
    """
    Test block for WebSearchService.

    TODO: Implement test cases
    Steps:
    1. Print header:
       print("=" * 70)
       print("WEB SEARCH SERVICE TEST")
       print("=" * 70)

    2. Test 1: Service initialization
       - Print: "Test 1: Service Initialization"
       - Create: service = WebSearchService(provider="duckduckgo")
       - Print provider and max_results
       - Print: "[OK] Service initialized"

    3. Test 2: Query validation
       - Print: "Test 2: Query Validation"
       - Test valid query: "test query"
       - Test too short query: "ab" (should raise ValueError)
       - Test long query: "a" * 600 (should truncate)
       - Print: "[OK] Query validation working"

    4. Test 3: DuckDuckGo search
       - Print: "Test 3: DuckDuckGo Search"
       - query = "Python programming language"
       - results = service.search(query, max_results=3)
       - Print number of results
       - For each result, print title and source
       - Print: "[OK] DuckDuckGo search working"

    5. Test 4: Format for agent
       - Print: "Test 4: Format for Agent"
       - formatted = service.format_for_agent(results, query)
       - Print formatted output
       - Print: "[OK] Formatting working"

    6. Test 5: Real-world query (EXTERNAL scenario)
       - Print: "Test 5: Real-world Query"
       - query = "current US inflation rate 2024"
       - results = service.search(query, max_results=5)
       - Print formatted results
       - Print: "[OK] Real-world query working"

    7. Print footer:
       print("=" * 70)
       print("ALL TESTS COMPLETE!")
       print("=" * 70)
    """
    print("=" * 70)
    print("WEB SEARCH SERVICE TEST")
    print("=" * 70)
    print("\n[TODO] Implement WebSearchService methods first")
    print("Then run: python -m src.services.web_search")
    print("=" * 70)
```

---

## 🏗️ Part 4: Implementation Details

### 4.1: DuckDuckGo Search Response Format

The `duckduckgo-search` package returns results in this format:

```python
[
    {
        "title": "Page Title",
        "href": "https://example.com/page",
        "body": "Snippet text describing the page content..."
    },
    ...
]
```

### 4.2: WebSearchResult Structure

```python
@dataclass
class WebSearchResult:
    title: str      # "Python Programming Language"
    url: str        # "https://python.org"
    snippet: str    # "Python is a programming language..."
    source: str     # "python.org"
```

### 4.3: Formatted Output Example

```
Web search results for: "current inflation rate"

1. US Inflation Rate - Latest Data
   Source: bls.gov
   URL: https://www.bls.gov/cpi/
   The Consumer Price Index shows current inflation trends...

2. Inflation News Today
   Source: reuters.com
   URL: https://reuters.com/markets/inflation
   Latest inflation reports indicate...

[Note: These results are from external web sources]
```

---

## 🧪 Part 5: Testing

### Test 1: Run standalone tests

```bash
cd backend
python -m src.services.web_search
```

### Test 2: Manual Python test

```python
from src.services.web_search import WebSearchService

service = WebSearchService()
results = service.search("Python programming", max_results=3)
for r in results:
    print(f"{r.title}: {r.url}")
```

### Test 3: Verify config

```python
from src.config.settings import Config

print(f"WEB_SEARCH_ENABLED: {Config.WEB_SEARCH_ENABLED}")
print(f"WEB_SEARCH_PROVIDER: {Config.WEB_SEARCH_PROVIDER}")
print(f"WEB_SEARCH_MAX_RESULTS: {Config.WEB_SEARCH_MAX_RESULTS}")
```

---

## ✅ Day 5 Checklist

### Part 1: Dependencies
- [ ] Install `duckduckgo-search` package
- [ ] Update `requirements.txt`

### Part 2: Configuration
- [ ] Add web search settings to `settings.py`
- [ ] Add settings to `.env`

### Part 3: WebSearchService
- [ ] Create `web_search.py` skeleton
- [ ] Implement `__init__()`
- [ ] Implement `_validate_query()`
- [ ] Implement `_extract_domain()`
- [ ] Implement `_search_duckduckgo()`
- [ ] Implement `format_for_agent()`
- [ ] (Optional) Implement `_search_tavily()`

### Part 4: Testing
- [ ] Run standalone tests
- [ ] Verify DuckDuckGo search returns results
- [ ] Verify formatting looks correct
- [ ] Test with real-world queries

---

## 🎯 Success Criteria

Day 5 is complete when:

- [ ] `WebSearchService` initializes without errors
- [ ] `search()` returns results from DuckDuckGo
- [ ] `format_for_agent()` produces readable output
- [ ] All standalone tests pass
- [ ] Config settings work correctly

---

## ⚠️ Common Issues

### Issue: DuckDuckGo rate limiting

**Solution**: Add delay between requests or reduce max_results

```python
import time
time.sleep(1)  # Add 1 second delay between searches
```

### Issue: No results returned

**Check**:
1. Query is not too specific
2. Internet connection is working
3. DuckDuckGo is not blocked

### Issue: Import error for duckduckgo_search

**Solution**: Install package
```bash
pip install duckduckgo-search
```

---

## 📚 References

- [DuckDuckGo Search Package](https://pypi.org/project/duckduckgo-search/)
- [Tavily API Docs](https://docs.tavily.com/)
- Week 3 Overview: [WEEK3_OVERVIEW.md](./WEEK3_OVERVIEW.md)
- Agent Tools: [../src/services/agent_tools.py](../src/services/agent_tools.py)

---

## 🚀 Next: Day 6

After completing Day 5, proceed to Day 6:
- Add `web_search` tool to agent
- Handle EXTERNAL recommendation in agent_tools.py
- End-to-end testing

See: [WEEK3_DAY6_IMPLEMENTATION.md](./WEEK3_DAY6_IMPLEMENTATION.md)
