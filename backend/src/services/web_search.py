"""
Web search service for external information retrieval.

Provides fallback search when internal documents don't contain
the requested information (EXTERNAL recommendation).

Week 3 - Day 5: Web Search Service
"""

from typing import Optional, List
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
