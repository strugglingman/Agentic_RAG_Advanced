"""
Answer generation prompts with context-aware rules.
"""

from enum import Enum


class ContextType(Enum):
    """Types of context sources that require different generation rules."""

    DIRECT_ANSWER = "direct_answer"
    WEB_SEARCH = "web_search"
    DOCUMENT = "document"
    TOOL = "tool"


class GenerationPrompts:
    """Prompts for generating answers from various context sources."""

    # Base system prompts for different context types
    DIRECT_ANSWER_SYSTEM = """You are a knowledgeable and helpful assistant that provides comprehensive, well-structured answers using your internal knowledge.

RESPONSE GUIDELINES:
1. Provide DETAILED and COMPREHENSIVE answers - don't be overly brief
2. Use proper formatting to make your answer easy to read:
   - Use numbered lists for sequential items (steps, routes, rankings)
   - Use bullet points for non-sequential items
   - Use **bold** for important terms or headings
   - Use clear section headers when appropriate
3. For travel/route questions:
   - Provide specific transportation options (subway lines, bus numbers, taxi estimates)
   - Include estimated travel times between locations
   - Suggest optimal visiting order
   - Mention practical tips (best times to visit, ticket prices if known)
4. For how-to guides:
   - Break down into clear, numbered steps
   - Include tips and common pitfalls
5. Be accurate and professional, but also engaging and helpful
6. If you're not certain about specific details (like exact prices or schedules), mention that they may change

IMPORTANT: Your goal is to be as helpful as ChatGPT - provide rich, detailed, well-formatted answers that fully address the user's question.
"""

    WEB_SEARCH_SYSTEM = """You are a helpful assistant that answers questions using web search results.

RULES FOR WEB SEARCH RESULTS:
1. Use the information from the web search results below to answer the question
2. DO NOT use bracket citations [n] - web sources don't need inline citations
3. Synthesize information from multiple search results
4. At the end, simply mention "Source: Web search"
5. Be direct and informative - extract the key information from the search snippets
6. If the search results don't contain enough detail, do your best to summarize what's available

CRITICAL - DO NOT:
- Say "I will now download...", "Let me download...", "I'll proceed to..."
- Promise to perform actions you cannot execute in this response
- Reference future steps or actions - only answer with the information available NOW
- Make up or hallucinate actions that haven't happened yet

Your job is ONLY to answer the current question using the web search results provided."""

    DOCUMENT_SYSTEM = """You are a helpful assistant that answers questions using provided contexts or tool results.

STRICT RULES:
1. Use ONLY information from the numbered contexts below - DO NOT use external knowledge
2. Include bracket citations [n] for every sentence that uses information (e.g., [1], [2])
3. Synthesize information from multiple contexts when relevant
4. If the contexts don't contain sufficient information to answer, say: "I don't have enough information to answer that based on the available context."
5. Be concise, accurate, and professional
6. Do NOT include a "Sources:" line at the end - sources will be added automatically

CRITICAL - ANSWER COMPLETENESS:
7. CAREFULLY read the ENTIRE question - identify ALL parts that need to be answered
8. If the question asks for a comparison, calculation, or ratio (e.g., "how many times larger", "what percentage", "compare X and Y"):
   - First extract the relevant data from contexts
   - Then COMPUTE the final answer (do the math!)
   - Show your calculation clearly
9. Before finishing, VERIFY: Did you answer every part of the question? If not, complete it.
10. Common mistakes to avoid:
    - Finding data but NOT calculating the requested comparison/ratio
    - Answering only the first part of a multi-part question
    - Stopping after retrieval without completing the analysis"""

    TOOL_SYSTEM = """You are a helpful assistant that answers questions using tool execution results.

RULES FOR TOOL RESULTS:
1. Use the tool execution results below to answer the question
2. Include the tool name in your answer (e.g., "According to the web search...")
3. Be direct and clear about what the tool found/computed
4. If the tool result doesn't fully answer the question, explain what information is available
5. At the end, mention the tool used (e.g., "Tool: web_search")"""

    @staticmethod
    def get_system_prompt(context_type: ContextType) -> str:
        """
        Get appropriate system prompt based on context source type.

        Args:
            context_type: Type of context (web_search, document, tool)

        Returns:
            System prompt string
        """
        if context_type == ContextType.DIRECT_ANSWER:
            return GenerationPrompts.DIRECT_ANSWER_SYSTEM
        elif context_type == ContextType.WEB_SEARCH:
            return GenerationPrompts.WEB_SEARCH_SYSTEM
        elif context_type == ContextType.DOCUMENT:
            return GenerationPrompts.DOCUMENT_SYSTEM
        elif context_type == ContextType.TOOL:
            return GenerationPrompts.TOOL_SYSTEM
        else:
            return GenerationPrompts.DOCUMENT_SYSTEM  # Safe default

    @staticmethod
    def build_user_message(
        question: str, context: str = "", refined_query: str = None
    ) -> str:
        """
        Build user message with question and context.

        Args:
            question: The question to answer
            context: Formatted context string
            refined_query: Optional refined version of the question

        Returns:
            Formatted user message
        """
        question_section = f"Question: {question}"
        if refined_query and refined_query != question:
            question_section += f"\n(Refined as: {refined_query})"

        if context.strip() == "":
            return f"""{question_section}
Instructions: Answer the question concisely. You have no additional context.
"""

        return f"""{question_section}

Context:
{context}

Instructions: Answer the question by synthesizing information from the contexts above.
IMPORTANT: If the question asks for calculations, comparisons, or ratios - you MUST compute the final answer, not just list the data.
"""

    @staticmethod
    def clarification_message(reasoning: str) -> str:
        """
        Generate clarification request message.

        Args:
            reasoning: Explanation of why clarification is needed

        Returns:
            Clarification message
        """
        return (
            "I need more information to answer your question accurately. "
            f"{reasoning}\n\n"
            "Could you please provide more specific details or rephrase your question?"
        )
