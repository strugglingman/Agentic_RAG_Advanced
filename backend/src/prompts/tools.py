"""
Tool calling prompts for calculator, web search, code execution, and other tools.
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
    def download_file_prompt(task: str, previous_step_context: str = "") -> str:
        """
        Generate prompt for download_file tool calling.

        Args:
            task: The download task description with URLs
            previous_step_context: Results from previous steps (e.g., web_search URLs)

        Returns:
            Prompt string for download_file function calling
        """
        context_section = ""
        if previous_step_context:
            context_section = f"""

📋 PREVIOUS STEP RESULTS (extract URLs from here):
{previous_step_context}

IMPORTANT: The URLs to download are in the previous step results above.
Extract ALL http:// or https:// URLs from the previous step results."""

        return f"""You need to call the download_file tool for this specific task.

Task: {task}{context_section}

INSTRUCTIONS:
1. Extract ALL file URLs from either:
   - The task description above (if URLs are explicitly listed)
   - The PREVIOUS STEP RESULTS section (if URLs came from web_search)
2. Look for:
   - http:// or https:// URLs
   - /api/files/ internal links
3. Return all URLs as a list in file_urls parameter

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
        previous_step_context: str = None,
    ) -> str:
        """
        Generate prompt for send_email tool calling.

        Args:
            task: The email task description
            available_file_ids: File IDs from previous tool results (download_file, create_documents)
            available_files: User's existing files from FileRegistry
            previous_step_context: Results from previous plan steps (contains file_ids created)

        Returns:
            Prompt string for send_email function calling
        """
        files_section = ""

        # Show previous step results (contains file_ids from create_documents, etc.)
        if previous_step_context:
            files_section += f"""

📋 PREVIOUS STEP RESULTS (use file_ids mentioned here):
{previous_step_context}"""

        if available_file_ids:
            files_section += f"""

⭐ FILES CREATED IN THIS SESSION (PRIORITY - use these file_ids for attachments):
{', '.join(available_file_ids)}
IMPORTANT: Use ONLY these file_ids for attachments, NOT file_ids from conversation history."""

        if available_files:
            file_list = []
            for f in available_files:
                file_list.append(
                    f"- {f.get('name')} (file_id: {f.get('file_id')}, category: {f.get('category')})"
                )
            files_section += f"""

USER'S AVAILABLE FILES:
{chr(10).join(file_list)}"""

        return f"""You need to call the send_email tool for this task.

Task: {task}{files_section}

IMPORTANT: The system has a Human-in-the-Loop (HITL) confirmation mechanism.
You should proceed to call the send_email tool - the user will be prompted to confirm
before the email is actually sent. DO NOT ask for confirmation in your response.

EMAIL ADDRESS VALIDATION:
- If a specific email address is provided in the task, use it
- If the email looks like a placeholder (e.g., user@example.com, test@test.com), still proceed - the HITL dialog will show it to the user
- If no email address is mentioned at all, extract it from the task context or use any email mentioned in the conversation

INSTRUCTIONS:
1. Extract recipient email address(es) from the task
2. Determine an appropriate email subject based on the task
3. Compose a professional email body
4. For attachments: Include ALL file_ids from "FILES CREATED IN THIS SESSION" section above (⭐ marked).
   - If multiple files were created/downloaded, attach ALL of them unless the user specifically requested only certain files.
   - DO NOT use file_ids from conversation history - those may be outdated.

CALL THE TOOL - the system will handle user confirmation before sending."""

    @staticmethod
    def code_execution_prompt(
        task: str,
        previous_step_context: str = "",
        is_detour: bool = False,
    ) -> str:
        """
        Generate prompt for code_execution tool calling.

        Args:
            task: The code execution task description
            previous_step_context: Data/results from previous steps to use in code
            is_detour: Whether this is an ad-hoc call (True) or planned (False)

        Returns:
            Prompt string for code_execution function calling
        """
        context_section = ""
        if previous_step_context:
            context_section = f"""

📋 DATA PROVIDED (use this data directly in your code):
{previous_step_context}

⭐ HOW TO USE THIS DATA:
- For CSV/tabular data: Use pd.read_csv(io.StringIO(data_string)) where data_string contains the CSV text
- Copy the data EXACTLY as shown above into a triple-quoted string variable
- Do NOT use pd.read_csv('filename.csv') - the file does NOT exist in the sandbox
- Use ACTUAL newlines in your string, not escaped \\n characters"""

        detour_note = ""
        if is_detour:
            detour_note = (
                "\n\nNote: This is a supplementary analysis based on previous results."
            )

        return f"""You need to call the code_execution tool for this task.

Task: {task}{context_section}{detour_note}

INSTRUCTIONS:
1. Generate Python code to accomplish the task
2. The last expression value will be captured as output (like Jupyter)
3. Embed data directly in the code (do NOT read from files unless specifically needed)
4. Available libraries: pandas, numpy, matplotlib, math, json, datetime
5. For multi-line strings (like CSV data), use triple quotes with ACTUAL newlines, NOT escaped \\n

⚠️ IMPORTANT - DO NOT CREATE FILES:
- Do NOT use to_excel(), to_csv(), to_json(), open(), or any file-writing operations
- Do NOT save results to files - the sandbox files are NOT accessible to users
- Instead, just COMPUTE the data and return it as the last expression
- If the user wants a file (Excel, PDF, CSV), a separate create_documents tool will handle that
- Your job is ONLY to compute/analyze data, not to create files

Example - WRONG:
```python
df.to_excel('output.xlsx')  # DON'T DO THIS
```

Example - CORRECT (embed data directly):
```python
import pandas as pd
from io import StringIO

# Embed the CSV data with ACTUAL newlines (not escaped \\n)
csv_data = '''name,status,hours
Alice,process,30
Bob,process,25
'''
df = pd.read_csv(StringIO(csv_data))
result = df.groupby('name')['hours'].sum()
result.to_dict()  # Return computed data as last expression
```

Call the code_execution tool with your Python code."""

    @staticmethod
    def fallback_prompt(query: str, tool_type: str) -> str:
        """
        Fallback prompt when no specific task is available.

        Args:
            query: User's original query
            tool_type: Type of tool ("calculator", "web_search", "download_file", "send_email", "create_documents", "code_execution")

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
        elif tool_type == "code_execution":
            return f"""Based on the user's query, generate and execute Python code.

User Query: {query}

Generate Python code to accomplish this task. Use print() to output results.
Available libraries: pandas, numpy, matplotlib, math, json, datetime.

Call the code_execution tool with your Python code."""
        else:
            return f"Use the appropriate tool to help answer this query: {query}"
