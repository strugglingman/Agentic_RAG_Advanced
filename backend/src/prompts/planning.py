"""
Planning prompts for query decomposition and step creation.
"""

from datetime import datetime, timezone
from typing import List, Optional


class PlanningPrompts:
    """Prompts for creating execution plans from user queries."""

    # File count limits per category (matching agent_service.py)
    CHAT_FILES_LIMIT = 10
    UPLOADED_FILES_LIMIT = 15
    DOWNLOADED_FILES_LIMIT = 10
    CREATED_FILES_LIMIT = 10

    @staticmethod
    def _format_time_ago(dt) -> str:
        """Format datetime as human-readable time ago string."""
        if not dt:
            return ""
        try:
            if isinstance(dt, str):
                # Parse ISO format string
                dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            diff = now - dt
            if diff.days > 30:
                return f"{diff.days // 30}mo ago"
            elif diff.days > 0:
                return f"{diff.days}d ago"
            elif diff.seconds > 3600:
                return f"{diff.seconds // 3600}h ago"
            elif diff.seconds > 60:
                return f"{diff.seconds // 60}m ago"
            else:
                return "just now"
        except Exception:
            return ""

    @staticmethod
    def _build_files_section(
        available_files: List[dict],
        attachment_file_ids: Optional[List[dict]] = None,
    ) -> str:
        """
        Build file context section for planning prompt.

        Groups files by category and formats them with file_id prominently displayed
        for use in email attachments and file operations.

        Matches agent_service.py logic:
        - Same categories: chat, uploaded, downloaded, created
        - Same order of display
        - Same file count limits per category
        - Shows file_id, original_name, download_url, and created_at

        Args:
            available_files: List of files from FileRegistry
            attachment_file_ids: Chat attachments uploaded in current message

        Returns:
            Formatted string section for files context
        """
        if not available_files and not attachment_file_ids:
            return ""

        sections = []

        # Current message attachments
        # Content is extracted by tool nodes when needed (e.g., create_documents)
        if attachment_file_ids:
            attachment_lines = []
            for f in attachment_file_ids:
                file_id = f.get("file_id", "")
                filename = f.get("filename", "unknown")
                mime_type = f.get("mime_type", "")
                attachment_lines.append(
                    f"  - **{filename}** (file_id: `{file_id}`, type: {mime_type})"
                )
            if attachment_lines:
                sections.append(
                    "### Current Message Attachments (⚠️ NOT searchable via retrieve!)\n"
                    "These files were uploaded with this message. Content will be passed directly to the tool.\n"
                    "⛔ NEVER use `retrieve` for these files - they are NOT in the RAG index!\n"
                    "✅ For DATA ANALYSIS (filter, aggregate, calculate): Use `code_execution` tool\n"
                    "✅ For FORMATTING into documents (PDF, DOCX): Use `create_documents` tool\n"
                    + "\n".join(attachment_lines)
                )

        if available_files:
            # Group by category (same as agent_service.py)
            by_category = {"chat": [], "uploaded": [], "downloaded": [], "created": []}
            for f in available_files:
                cat = f.get("category", "uploaded")
                if cat in by_category:
                    by_category[cat].append(f)

            # Chat attachments (from previous messages in conversation)
            # Limit to CHAT_FILES_LIMIT most recent
            if by_category["chat"]:
                lines = []
                for f in by_category["chat"][: PlanningPrompts.CHAT_FILES_LIMIT]:
                    file_id = f.get("id", "")
                    name = f.get("original_name", "unknown")
                    url = f.get("download_url", f"/api/files/{file_id}")
                    time_ago = PlanningPrompts._format_time_ago(f.get("created_at"))
                    time_info = f", {time_ago}" if time_ago else ""
                    lines.append(
                        f"  - **{name}** (file_id: `{file_id}`{time_info}) → [Download]({url})"
                    )
                sections.append(
                    "### Chat Attachments (previous messages)\n" + "\n".join(lines)
                )

            # Uploaded files (RAG indexed)
            # Limit to UPLOADED_FILES_LIMIT most recent
            if by_category["uploaded"]:
                lines = []
                for f in by_category["uploaded"][
                    : PlanningPrompts.UPLOADED_FILES_LIMIT
                ]:
                    file_id = f.get("id", "")
                    name = f.get("original_name", "unknown")
                    url = f.get("download_url", f"/api/files/{file_id}")
                    indexed = f.get("indexed_in_chromadb", False)
                    rag_status = "✓ RAG indexed" if indexed else "not indexed"
                    time_ago = PlanningPrompts._format_time_ago(f.get("created_at"))
                    time_info = f", {time_ago}" if time_ago else ""
                    lines.append(
                        f"  - **{name}** (file_id: `{file_id}`, {rag_status}{time_info}) → [Download]({url})"
                    )
                sections.append(
                    "### Uploaded Files (RAG documents)\n" + "\n".join(lines)
                )

            # Downloaded files
            # Limit to DOWNLOADED_FILES_LIMIT most recent
            # Downloaded files are NOT indexed in ChromaDB - cannot use retrieve on them
            if by_category["downloaded"]:
                lines = []
                for f in by_category["downloaded"][
                    : PlanningPrompts.DOWNLOADED_FILES_LIMIT
                ]:
                    file_id = f.get("id", "")
                    name = f.get("original_name", "unknown")
                    url = f.get("download_url", f"/api/files/{file_id}")
                    source = f.get("source_url", "")
                    source_info = f" from {source}" if source else ""
                    lines.append(
                        f"  - **{name}** (file_id: `{file_id}`, ⚠ NOT searchable){source_info} → [Download]({url})"
                    )
                sections.append(
                    "### Downloaded Files (for attachments only, NOT searchable via retrieve)\n"
                    + "\n".join(lines)
                )

            # Created files
            # Limit to CREATED_FILES_LIMIT most recent
            # Created files are NOT indexed in ChromaDB - cannot use retrieve on them
            if by_category["created"]:
                lines = []
                for f in by_category["created"][: PlanningPrompts.CREATED_FILES_LIMIT]:
                    file_id = f.get("id", "")
                    name = f.get("original_name", "unknown")
                    url = f.get("download_url", f"/api/files/{file_id}")
                    lines.append(
                        f"  - **{name}** (file_id: `{file_id}`, ⚠ NOT searchable) → [Download]({url})"
                    )
                sections.append(
                    "### Created Documents (for attachments only, NOT searchable via retrieve)\n"
                    + "\n".join(lines)
                )

        if not sections:
            return ""

        return (
            "\n## Available Files\n"
            "- Use file_id (e.g., `cmjg8yrab0000xspw5ip79qcb`) when referencing files for email attachments.\n"
            "- Use download_url (e.g., `/api/files/{file_id}`) when providing download links to user.\n"
            "- **IMPORTANT**: Only files with `✓ RAG indexed` can be searched via `retrieve`. Files with `⚠ NOT searchable` are for download/attachment only.\n\n"
            + "\n\n".join(sections)
            + "\n"
        )

    @staticmethod
    def create_plan(
        query: str,
        conversation_context: str = "",
        available_files: Optional[List[dict]] = None,
        attachment_file_ids: Optional[List[dict]] = None,
    ) -> str:
        """
        Generate prompt for creating a multi-step execution plan.

        Args:
            query: User's original question
            conversation_context: Summary of recent conversation for context understanding
            available_files: List of user's files from FileRegistry
            attachment_file_ids: Chat attachments uploaded in current message

        Returns:
            Prompt string for plan generation
        """
        context_section = ""
        if conversation_context:
            context_section = f"""
## Recent Conversation Context
{conversation_context}

Use conversation context ONLY when user explicitly references it with words like:
- Chinese: "这些", "那些", "以上", "之前", "刚才说的", "上面的"
- English: "these", "those", "it", "them", "the above", "what you said"

For example: "这些景点的路线" → include actual attraction names from context.
For standalone queries (no reference words), create a fresh plan based on the current query only.
"""

        files_section = PlanningPrompts._build_files_section(
            available_files or [], attachment_file_ids
        )

        return f"""
You are a planning assistant. Create a plan to answer this query using available tools.
{context_section}{files_section}
## User Query
{query}

## Available Tools (use ONLY these exact tool names):
- direct_answer: Answer using LLM's built-in knowledge (for general knowledge, travel routes, cultural info, how-to guides, explanations, simple math)
- retrieve: Search internal COMPANY documents and uploaded files ONLY
- web_search: Search the web for CURRENT/real-time information (weather, news, stock prices, recent events)
- download_file: Download a file from a URL and save to user's storage (returns file_id for chaining)
- send_email: Send an email with optional file attachments (use file_id from available files or from download_file/create_documents output)
- create_documents: Create documents (PDF, DOCX, TXT, CSV, XLSX, HTML, MD) from content (returns file_id for chaining)
- code_execution: Execute Python code in a secure sandbox for data analysis, complex calculations, statistics, and file processing

## CRITICAL TOOL SELECTION RULES:

### Use "direct_answer" for:
- General knowledge (history, geography, science, culture)
- Travel recommendations, routes, and itineraries
- How-to guides and explanations
- Information about famous places, people, books
- Anything the LLM already knows from training

### Use "retrieve" ONLY for:
- Files marked with "✓ RAG indexed" in the available files list
- Company internal documents that have been indexed
- User-uploaded files that are searchable
- IMPORTANT: Files marked "⚠ NOT searchable" CANNOT be searched via retrieve - use direct_answer or web_search instead

### Use "web_search" ONLY for:
- Current/real-time information (today's weather, live prices)
- Very recent events (within last few months)
- Information that changes frequently

### Use "download_file" for:
- Downloading files from external URLs
- Saving web resources for later use or email attachment
- Output includes file_id which can be used with send_email

### Use "send_email" for:
- Sending emails to specified recipients
- Attaching files using file_id (from available files list or tool outputs)
- IMPORTANT: Use exact file_id values, do not modify them
- CRITICAL: ONLY add send_email step if the CURRENT USER QUERY explicitly requests to email AND provides a specific email address IN THE CURRENT QUERY
- DO NOT use email addresses from conversation history or previous messages - only from the CURRENT query
- DO NOT add send_email step just because a document was created - user may only want the download link
- If user says "download files for me" without mentioning email, do NOT add send_email

### Use "create_documents" for:
- FORMATTING already-known content into document files (PDF, DOCX, TXT, CSV, XLSX, HTML, MD)
- Output AUTOMATICALLY includes download link - NO separate step needed for "give me link" requests
- Output includes file_id which can be used with send_email
- IMPORTANT: When user asks to "create document and give link", use ONLY ONE step: create_documents (the link is included in output)
- ⚠️ WARNING: create_documents ONLY FORMATS text - it does NOT compute, filter, or aggregate data!
- ⚠️ For ANY data analysis task, use code_execution FIRST, then optionally create_documents for output

### Use "code_execution" for (MUST USE for data analysis):
- ⭐ ANALYZING uploaded files (CSV, Excel, JSON) - filtering, aggregating, summarizing data
- ⭐ ANY task involving "analyze", "calculate", "sum", "average", "filter", "count" on file data
- Complex calculations that require programming logic
- Statistical analysis (mean, median, standard deviation, correlations)
- Data transformation and processing (Excel/CSV data manipulation)
- Mathematical operations beyond simple arithmetic
- Generating charts or visualizations
- Available libraries: pandas, numpy, matplotlib, math, json, datetime
- ⚠️ CRITICAL: If user uploads a file and asks to "analyze", "summarize data", "calculate totals" → MUST use code_execution, NOT create_documents

## FILE OPERATIONS - IMPORTANT:
- When user asks to email a file, use the file_id from the Available Files section above
- When user asks to download something and email it, chain: download_file → send_email (use file_id from download_file output)
- When user asks to create a document and email it, chain: create_documents → send_email (use file_id from create_documents output)
- Inline attachment content (marked [INLINE CONTENT AVAILABLE]) has already been extracted - no need to retrieve it

## ⭐ FILE ANALYSIS vs FILE CREATION - CRITICAL DISTINCTION:
- **ANALYZE/COMPUTE data** (filter, sum, average, count, statistics) → Use `code_execution`
- **FORMAT/SAVE content** (create PDF, DOCX, export) → Use `create_documents`
- **ANALYZE then EXPORT**: Use `code_execution` FIRST, then `create_documents` for output
- Keywords that REQUIRE code_execution: "analyze", "calculate", "sum", "total", "average", "filter", "count", "statistics", "aggregate"
- Keywords for create_documents: "create document", "export as PDF", "save as file", "generate report" (only if content is already known)

## CRITICAL: MULTI-STEP vs SINGLE-STEP PLANS

**IMPORTANT: Detect these keywords for multi-step plans:**
- "download" / "下载" → Add download_file step after web_search
- "email" / "send to" / "发送" → Add send_email step at the end
- "create document" / "generate file" / "整理成文件" → Add create_documents step

**Use MULTI-STEP plans when steps have TRUE DEPENDENCIES (step 2 needs output from step 1):**
- ✅ download_file → send_email (must download before attaching)
- ✅ create_documents → send_email (must create before attaching)
- ✅ retrieve → code_execution (get data, then do complex analysis/statistics)
- ✅ code_execution → create_documents (analyze data, then format as document)
- ✅ web_search → create_documents (search info, then create report)
- ✅ web_search → download_file (search for resources, then download the URLs found)
- ✅ web_search → download_file → send_email (search, download, then email the downloaded files)

**CRITICAL: web_search returns URLs, NOT files. To send files from web, you MUST:**
1. web_search: Find the resources/URLs
2. download_file: Download the URLs to get file_ids
3. send_email: Attach files using the file_ids from download_file

**NEVER create "fallback" plans where steps are ALTERNATIVES (try X, if fail try Y):**
- ❌ ["retrieve", "web_search"] ← WRONG! System auto-fallbacks to web_search if retrieve fails
- ❌ ["retrieve", "direct_answer"] ← WRONG! These are alternatives, not dependencies
- ❌ ["retrieve", "web_search", "direct_answer"] ← WRONG! Redundant fallback chain

**The system has AUTOMATIC fallback handling:**
- If "retrieve" finds insufficient results → system AUTOMATICALLY routes to "web_search"
- You only need to specify the PRIMARY tool for the query

**Use SINGLE-STEP for simple queries:**
- ✅ ["retrieve: A Man Called Ove"] ← Auto-fallback handles web_search if needed
- ✅ ["web_search: Beijing weather"] ← Single step for real-time info
- ✅ ["direct_answer: Explain quantum physics"] ← Single step for general knowledge

## IMPORTANT:
1. For **retrieve**: Write SHORT keyword-based queries (not instructions). Example: "A Man Called Ove character Ove Sonja" NOT "Search files for content about..."
2. For **direct_answer/web_search**: Write detailed, comprehensive queries describing what answer is needed
3. For travel/route questions, specify: locations, order, and request for transportation details
4. The LLM will generate a detailed, well-formatted answer based on your plan
5. **Multi-step = dependencies, Single-step = alternatives** (system handles fallbacks)

## Examples:
- Query: "南京的10个旅游景点" → {{"steps": ["direct_answer: List and describe the top 10 tourist attractions in Nanjing, China, including historical significance and visitor tips"]}}
- Query: "这些景点的具体路线" (after discussing Nanjing attractions) → {{"steps": ["direct_answer: Provide a detailed travel itinerary and route connecting Nanjing Memorial Hall, Zhongshan Mausoleum, Ming Xiaoling Tomb, Xuanwu Lake, Yangtze River Bridge, Confucius Temple, Nanjing Museum, Zhonghua Gate, Purple Mountain Observatory, and Jiming Temple, including transportation options between each location, recommended visiting order, and time estimates"]}}
- Query: "What is our Q3 revenue?" → {{"steps": ["retrieve: Q3 quarterly revenue report"]}}
- Query: "Tell me about the man called Ove" → {{"steps": ["retrieve: A Man Called Ove Ove character personality Sonja"]}}
- Query: "明天北京天气怎么样" → {{"steps": ["web_search: Beijing weather forecast tomorrow"]}}
- Query: "Download the report from example.com/report.pdf and email it to john@company.com" → {{"steps": ["download_file: https://example.com/report.pdf", "send_email: Send email to john@company.com with attached file (use file_id from previous step)"]}}
- Query: "Email me the Q3 report" (with Q3_Report.pdf in available files) → {{"steps": ["send_email: Send Q3_Report.pdf (file_id: cmjg8yrab0000xspw5ip79qcb) to user"]}}
- Query: "Create a summary document and send it to my manager" → {{"steps": ["create_documents: Create PDF summary document with key points", "send_email: Send the created document to manager (use file_id from previous step)"]}}
- Query: "帮我整理成文件给我链接" (create document and give me link) → {{"steps": ["create_documents: Create PDF document with the requested content"]}} (ONLY ONE step - link is auto-included in output)
- Query: "Create a report and provide download link" → {{"steps": ["create_documents: Create the report document"]}} (ONLY ONE step - link is auto-included)
- Query: "Find articles about Nanjing travel and send them to john@example.com" → {{"steps": ["web_search: Find travel articles and guides about Nanjing tourism", "download_file: Download the web pages from the URLs found in web search", "send_email: Send the downloaded files to john@example.com with attached files (use file_ids from download_file)"]}}
- Query: "Search for Python tutorials and download them for me" → {{"steps": ["web_search: Find Python programming tutorials and guides", "download_file: Download the tutorial pages from the URLs found"]}}
- Query: "show me places to visit in Fuzhou and download the files" → {{"steps": ["web_search: Top tourist attractions in Fuzhou China with descriptions and links", "download_file: Download the web pages from the URLs found in web search results"]}}
- Query: "Analyze this CSV file and summarize total sales" (with sales.csv attached) → {{"steps": ["code_execution: Read the CSV file, calculate total sales, and print summary statistics"]}}
- Query: "Analyze employees.csv, filter status=process, sum hours" (with employees.csv attached) → {{"steps": ["code_execution: Read employees.csv, filter rows where status='process', calculate sum of hours column"]}}
- Query: "Analyze the data and create a PDF report" (with data.csv attached) → {{"steps": ["code_execution: Read and analyze the CSV data, compute statistics and summaries", "create_documents: Create PDF report with the analysis results from code_execution"]}}

Return ONLY JSON:
{{"steps": ["tool_name: detailed comprehensive query with full context", ...]}}
"""
