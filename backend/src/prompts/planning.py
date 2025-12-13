"""
Planning prompts for query decomposition and step creation.
"""

from typing import Dict, Any


class PlanningPrompts:
    """Prompts for creating execution plans from user queries."""

    @staticmethod
    def create_plan(query: str) -> str:
        """
        Generate prompt for creating a multi-step execution plan.

        Args:
            query: User's original question

        Returns:
            Prompt string for plan generation
        """
        return f"""
You are a planning assistant. Create a minimal plan to answer this query using available tools.

Query: {query}

Available Tools (use ONLY these exact tool names):
- retrieve: Search internal documents and knowledge base
- calculator: Perform mathematical calculations
- web_search: Search the web for external/current information

IMPORTANT RULES:
1. ONLY create steps that call a tool - no "review", "summarize", "format" steps
2. Use exact tool names: "retrieve", "calculator", "web_search"
3. Each step MUST start with one of the tool names
4. Keep plan minimal - 1-3 steps maximum
5. The system will automatically generate the final answer after all tool calls
6. CRITICAL: After the colon, write an OPTIMIZED search query in English that will find the information
   - For book/product names: Use the proper English name (e.g., "Summarize the book 'The Man Called Ove'" not "介绍一下the man called ove这本书")
   - For locations: Use English transliteration (e.g., "Nanjing" not "南京")
   - For general terms: Translate to English keywords
   - Keep it concise and searchable

Examples of GOOD plans:
- Query: "What is our Q3 revenue?" → {{"steps": ["retrieve: Q3 revenue"]}}
- Query: "介绍一下the man called Ove这本书" → {{"steps": ["retrieve: Summarize the book 'The Man Called Ove'"]}}
- Query: "南京明天天气" → {{"steps": ["web_search: Nanjing weather tomorrow"]}}
- Query: "Calculate 15% of our budget" → {{"steps": ["retrieve: budget", "calculator: 15% of budget"]}}
- Query: "Compare our revenue to industry" → {{"steps": ["retrieve: company revenue", "web_search: industry revenue benchmarks"]}}

Examples of BAD plans (DO NOT DO THIS):
- {{"steps": ["retrieve: 介绍一下the man called ove这本书"]}} ← Don't copy Chinese query as-is, use "The Man Called Ove"
- {{"steps": ["retrieve: get data", "review the results"]}} ← "review" is NOT a tool
- {{"steps": ["web_search: find info", "format the response"]}} ← "format" is NOT a tool

Return ONLY JSON:
{{"steps": ["tool_name: optimized search query", ...]}}
"""
