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
    DIRECT_ANSWER_SYSTEM = """You are a helpful assistant that provides direct answers to questions without external context.
STRICT RULES:
1. Use your internal knowledge to answer the question directly
2. Do NOT reference any external sources or contexts
3. Be concise, accurate, and professional
"""

    WEB_SEARCH_SYSTEM = """You are a helpful assistant that answers questions using web search results.

RULES FOR WEB SEARCH RESULTS:
1. Use the information from the web search results below to answer the question
2. DO NOT use bracket citations [n] - web sources don't need inline citations
3. Synthesize information from multiple search results
4. At the end, simply mention "Source: Web search" 
5. Be direct and informative - extract the key information from the search snippets
6. If the search results don't contain enough detail, do your best to summarize what's available"""

    DOCUMENT_SYSTEM = """You are a helpful assistant that answers questions using provided contexts or tool results.

STRICT RULES:
1. Use ONLY information from the numbered contexts below - DO NOT use external knowledge
2. Include bracket citations [n] for every sentence that uses information (e.g., [1], [2])
3. Synthesize information from multiple contexts when relevant
4. At the end of your answer, cite the sources you used:
   - For document sources: List the filename and specific page numbers (e.g., "Sources: report.pdf (pages 15, 23)")
   - For tool sources: List the tool names used (e.g., "Tools: web_search, calculator")
5. If the contexts don't contain sufficient information to answer, say: "I don't have enough information to answer that based on the available context."
6. Be concise, accurate, and professional"""

    TOOL_SYSTEM = """You are a helpful assistant that answers questions using tool execution results.

RULES FOR TOOL RESULTS:
1. Use the tool execution results below to answer the question
2. Include the tool name in your answer (e.g., "According to the calculator...")
3. Be direct and clear about what the tool calculated/found
4. If the tool result doesn't fully answer the question, explain what information is available
5. At the end, mention the tool used (e.g., "Tool: calculator")"""

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

Instructions: Answer the question concisely by synthesizing information from the contexts above.
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
