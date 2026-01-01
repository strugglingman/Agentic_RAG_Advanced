"""
Tool calling prompts for calculator and web search.
"""


class ToolPrompts:
    """Prompts for LLM function calling with tools."""

    @staticmethod
    def calculator_prompt(task: str, is_detour: bool = False) -> str:
        """
        Generate prompt for calculator tool calling.

        Args:
            task: The calculation task description
            is_detour: Whether this is an ad-hoc call (True) or planned (False)

        Returns:
            Prompt string for calculator function calling
        """
        context_note = ""
        if is_detour:
            context_note = "\n\nNote: This is a supplementary calculation based on previous results."

        return f"""You need to call the calculator tool for this specific task.

Task: {task}

IMPORTANT: Extract the calculation request ONLY from the task description above. Do NOT use unrelated parts from other tasks.{context_note}

Call the calculator tool with the appropriate expression."""

    @staticmethod
    def web_search_prompt(task: str, is_detour: bool = False) -> str:
        """
        Generate prompt for web search tool calling.

        Args:
            task: The search task description
            is_detour: Whether this is an ad-hoc call (True) or planned (False)

        Returns:
            Prompt string for web_search function calling
        """
        context_note = ""
        if is_detour:
            context_note = (
                "\n\nNote: This is a supplementary search based on previous results."
            )

        return f"""You need to call the web_search tool for this specific task.

Task: {task}

IMPORTANT: Extract the search query ONLY from the task description above. Do NOT use unrelated parts from other tasks.{context_note}

Call the web_search tool with the appropriate query."""

    @staticmethod
    def download_file_prompt(task: str) -> str:
        """
        Generate prompt for download_file tool calling.

        Args:
            task: The download task description with URLs

        Returns:
            Prompt string for download_file function calling
        """
        return f"""You need to call the download_file tool for this specific task.

Task: {task}

IMPORTANT: Extract ALL file URLs from the task description above. Look for:
- http:// or https:// URLs
- /api/files/ internal links

Return all URLs as a list in file_urls parameter.

Call the download_file tool with the appropriate file_urls array."""

    @staticmethod
    def create_documents_prompt(task: str, previous_content: str = "") -> str:
        """
        Generate prompt for create_documents tool calling.

        Args:
            task: The document creation task description
            previous_content: Content from previous steps to include in document

        Returns:
            Prompt string for create_documents function calling
        """
        content_section = ""
        if previous_content:
            content_section = f"""

AVAILABLE CONTENT FROM PREVIOUS STEPS:
{previous_content}

You may use this content as the basis for the document."""

        return f"""You need to call the create_documents tool for this specific task.

Task: {task}{content_section}

INSTRUCTIONS:
1. Determine the appropriate format (pdf, docx, txt, md, html, csv, xlsx)
2. Create a meaningful title based on the task
3. Generate the content for the document
4. If previous step content is available, incorporate it appropriately

Call the create_documents tool with the documents array."""

    @staticmethod
    def send_email_prompt(
        task: str,
        available_file_ids: list = None,
        available_files: list = None,
    ) -> str:
        """
        Generate prompt for send_email tool calling.

        Args:
            task: The email task description
            available_file_ids: File IDs from previous tool results (download_file, create_documents)
            available_files: User's existing files from FileRegistry

        Returns:
            Prompt string for send_email function calling
        """
        files_section = ""

        if available_file_ids:
            files_section += f"""

FILES FROM PREVIOUS STEPS (use file_id for attachments):
{', '.join(available_file_ids)}"""

        if available_files:
            file_list = []
            for f in available_files:
                file_list.append(f"- {f.get('name')} (file_id: {f.get('file_id')}, category: {f.get('category')})")
            files_section += f"""

USER'S AVAILABLE FILES:
{chr(10).join(file_list)}"""

        return f"""You need to call the send_email tool for this specific task.

Task: {task}{files_section}

INSTRUCTIONS:
1. Extract recipient email addresses from the task
2. Determine the email subject
3. Compose the email body
4. For attachments, use the file_id values from the available files above

Call the send_email tool with to, subject, body, and optionally attachments (array of file_ids)."""

    @staticmethod
    def fallback_prompt(query: str, tool_type: str) -> str:
        """
        Fallback prompt when no specific task is available.

        Args:
            query: User's original query
            tool_type: Type of tool ("calculator", "web_search", "download_file", "send_email", "create_documents")

        Returns:
            Generic tool calling prompt
        """
        if tool_type == "calculator":
            return f"Based on the user's query, call the appropriate calculator tool.\n\nUser Query: {query}"
        elif tool_type == "web_search":
            return f"Based on the user's query, search the web for relevant information.\n\nUser Query: {query}"
        elif tool_type == "download_file":
            return f"Based on the user's query, download the requested files.\n\nUser Query: {query}"
        elif tool_type == "send_email":
            return f"Based on the user's query, send an email.\n\nUser Query: {query}"
        elif tool_type == "create_documents":
            return f"Based on the user's query, create the requested documents.\n\nUser Query: {query}"
        else:
            return f"Use the appropriate tool to help answer this query: {query}"
