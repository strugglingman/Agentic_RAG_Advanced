"""
Planning prompts for query decomposition and step creation.
"""


class PlanningPrompts:
    """Prompts for creating execution plans from user queries."""

    @staticmethod
    def create_plan(query: str, conversation_context: str = "") -> str:
        """
        Generate prompt for creating a multi-step execution plan.

        Args:
            query: User's original question
            conversation_context: Summary of recent conversation for context understanding

        Returns:
            Prompt string for plan generation
        """
        context_section = ""
        if conversation_context:
            context_section = f"""
## Recent Conversation Context
{conversation_context}

CRITICAL: The user may reference previous messages using words like "这些", "以上", "those", "these", "it", "them".
You MUST resolve these references using the conversation context above and include the ACTUAL content in your plan.
For example: If user previously discussed "南京10个景点" and now asks "这些景点的路线", you must include the actual attraction names.
"""

        return f"""
You are a planning assistant. Create a plan to answer this query using available tools.
{context_section}
## User Query
{query}

## Available Tools (use ONLY these exact tool names):
- direct_answer: Answer using LLM's built-in knowledge (for general knowledge, travel routes, cultural info, how-to guides, explanations)
- retrieve: Search internal COMPANY documents and uploaded files ONLY
- calculator: Perform mathematical calculations
- web_search: Search the web for CURRENT/real-time information (weather, news, stock prices, recent events)

## CRITICAL TOOL SELECTION RULES:

### Use "direct_answer" for:
- General knowledge (history, geography, science, culture)
- Travel recommendations, routes, and itineraries
- How-to guides and explanations
- Information about famous places, people, books
- Anything the LLM already knows from training

### Use "retrieve" ONLY for:
- Company internal documents
- User-uploaded files
- Organization-specific policies or data

### Use "web_search" ONLY for:
- Current/real-time information (today's weather, live prices)
- Very recent events (within last few months)
- Information that changes frequently

## IMPORTANT:
1. Write DETAILED, COMPREHENSIVE queries after the colon
2. Include ALL context from conversation when resolving references
3. For travel/route questions, specify: locations, order, and request for transportation details
4. The LLM will generate a detailed, well-formatted answer based on your plan

## Examples:
- Query: "南京的10个旅游景点" → {{"steps": ["direct_answer: List and describe the top 10 tourist attractions in Nanjing, China, including historical significance and visitor tips"]}}
- Query: "这些景点的具体路线" (after discussing Nanjing attractions) → {{"steps": ["direct_answer: Provide a detailed travel itinerary and route connecting Nanjing Memorial Hall, Zhongshan Mausoleum, Ming Xiaoling Tomb, Xuanwu Lake, Yangtze River Bridge, Confucius Temple, Nanjing Museum, Zhonghua Gate, Purple Mountain Observatory, and Jiming Temple, including transportation options between each location, recommended visiting order, and time estimates"]}}
- Query: "What is our Q3 revenue?" → {{"steps": ["retrieve: Q3 quarterly revenue report"]}}
- Query: "明天北京天气怎么样" → {{"steps": ["web_search: Beijing weather forecast tomorrow"]}}

Return ONLY JSON:
{{"steps": ["tool_name: detailed comprehensive query with full context", ...]}}
"""
