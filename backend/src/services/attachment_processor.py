"""
Attachment Processor - Pre-processes uploaded files for LLM understanding.

This module handles:
1. File content extraction (PDF, TXT, DOCX, etc.)
2. Image analysis via Vision API (or pass-through for native multimodal)
3. Building enriched attachment context for LLM

Architecture:
============
    User uploads file → Frontend sends base64 → AttachmentProcessor →
    Enriched context (with content/description) → LLM can reason about files

Key Concepts:
=============
- Images: Can pass directly to LLM via multimodal API (LLM "sees" them)
- Documents: Must extract text first (LLM can only read text strings)

Usage:
======
    processor = AttachmentProcessor(openai_client)
    enriched = processor.process_attachments(raw_attachments)
    context_str = processor.build_llm_context(enriched)
"""

from typing import Dict, Any, List, Optional, Protocol
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================


class AttachmentType(str, Enum):
    """
    Supported attachment types.

    TODO: Add more types as needed (e.g., AUDIO, VIDEO)
    """

    IMAGE = "image"
    PDF = "pdf"
    TEXT = "text"  # txt, csv, md
    DOCUMENT = "document"  # docx, doc
    SPREADSHEET = "spreadsheet"  # xlsx, xls
    UNKNOWN = "unknown"


@dataclass
class RawAttachment:
    """
    Raw attachment as received from frontend.

    This matches the payload structure sent by ChatUI.tsx:
    {
        type: "image" | "file",
        filename: "report.pdf",
        mime_type: "application/pdf",
        data: "base64encodedstring..."
    }
    """

    index: int  # Position for chat_attachment_N reference
    filename: str  # Original filename
    mime_type: str  # MIME type (e.g., "image/png", "application/pdf")
    data: str  # Base64 encoded content
    type: str  # "image" | "file" (from frontend)


@dataclass
class EnrichedAttachment:
    """
    Processed attachment with extracted content/description.

    After processing:
    - Images: Have 'description' (from Vision API) OR raw data for native multimodal
    - Documents: Have 'content' (extracted text)
    - Failed: Have 'error' message

    The LLM sees a summary like:
        "chat_attachment_0 [IMAGE] sales_chart.png: A bar chart showing Q3 revenue..."
        "chat_attachment_1 [PDF] report.pdf (5 pages): Executive summary discussing..."
    """

    index: int
    filename: str
    mime_type: str
    data: str  # Preserved for forwarding (email attachments)
    attachment_type: AttachmentType
    content: Optional[str] = None  # Extracted text (for documents)
    description: Optional[str] = None  # AI description (for images)
    error: Optional[str] = None  # Error message if processing failed
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )  # Extra info (page count, etc.)

    def to_llm_context(self, max_preview_chars: int = 500) -> str:
        """
        Format attachment for LLM context string.

        TODO: Implement this method
        =====
        Return format: "chat_attachment_N [TYPE] filename (metadata): preview..."

        Example outputs:
        - "chat_attachment_0 [IMAGE] chart.png: A bar chart showing sales trends..."
        - "chat_attachment_1 [PDF] report.pdf (5 pages): Executive summary of Q3..."
        - "chat_attachment_2 [TEXT] notes.txt: Meeting notes from Monday..."

        Steps:
        1. Build reference string: f"chat_attachment_{self.index}"
        2. Get type label: self.attachment_type.value.upper()
        3. Build metadata string from self.metadata (e.g., "5 pages", "1024x768")
        4. Get preview: use self.description (images) or self.content (docs)
        5. Truncate preview to max_preview_chars
        6. Handle error case: show "[Processing failed: {error}]"
        """
        raise NotImplementedError("TODO: Implement to_llm_context")


# ============================================================================
# PROCESSOR PROTOCOL (Interface)
# ============================================================================


class FileProcessor(Protocol):
    """
    Protocol (interface) for file type processors.

    Each processor handles specific file types (images, PDFs, etc.)
    Implement this protocol for each new file type you want to support.
    """

    def can_process(self, mime_type: str, filename: str) -> bool:
        """Check if this processor handles the given file type."""
        ...

    def process(
        self, attachment: RawAttachment, openai_client: Any
    ) -> EnrichedAttachment:
        """Process the attachment and return enriched version."""
        ...


# ============================================================================
# TYPE-SPECIFIC PROCESSORS
# ============================================================================


class ImageProcessor:
    """
    Process images - either via Vision API or pass-through for native multimodal.

    Supported formats: PNG, JPEG, GIF, WEBP

    Two strategies:
    - Option A: Call Vision API to get text description (works with any model)
    - Option B: Pass image data through, let message builder handle multimodal
    """

    SUPPORTED_MIMES = {"image/png", "image/jpeg", "image/gif", "image/webp"}

    def can_process(self, mime_type: str, filename: str) -> bool:
        """
        TODO: Check if mime_type is in SUPPORTED_MIMES
        """
        raise NotImplementedError("TODO: Implement can_process")

    def process(
        self, attachment: RawAttachment, openai_client: Any
    ) -> EnrichedAttachment:
        """
        Process image attachment.

        TODO: Implement this method
        =====
        Option A - Generate description via Vision API:
            1. Build multimodal message:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image concisely..."},
                        {"type": "image_url", "image_url": {
                            "url": f"data:{mime_type};base64,{data}"
                        }}
                    ]
                }]
            2. Call OpenAI API (model="gpt-4o-mini" or similar vision model)
            3. Extract description from response
            4. Return EnrichedAttachment with description set

        Option B - Pass-through for native multimodal:
            1. Just return EnrichedAttachment with data preserved
            2. Let _build_initial_messages handle the multimodal format
            3. Set description=None, no error

        Handle errors gracefully - return EnrichedAttachment with error set
        """
        raise NotImplementedError("TODO: Implement ImageProcessor.process")


class TextProcessor:
    """
    Process plain text files (txt, csv, md, log).

    Simple: decode base64 to UTF-8 string.
    """

    SUPPORTED_MIMES = {"text/plain", "text/csv", "text/markdown"}
    SUPPORTED_EXTENSIONS = {".txt", ".csv", ".md", ".log"}

    def can_process(self, mime_type: str, filename: str) -> bool:
        """
        TODO: Check if mime_type is in SUPPORTED_MIMES
              OR if filename extension is in SUPPORTED_EXTENSIONS

        Hint: Extract extension with filename.rsplit(".", 1)[-1].lower()
        """
        raise NotImplementedError("TODO: Implement can_process")

    def process(
        self, attachment: RawAttachment, openai_client: Any
    ) -> EnrichedAttachment:
        """
        Decode text content from base64.

        TODO: Implement this method
        =====
        Steps:
        1. Import base64 module
        2. Decode: content = base64.b64decode(attachment.data).decode('utf-8', errors='ignore')
        3. Truncate if too large (e.g., MAX_CHARS = 10000)
        4. Build metadata: {"lines": count, "chars": count, "truncated": bool}
        5. Return EnrichedAttachment with content set

        Handle exceptions - return EnrichedAttachment with error set
        """
        raise NotImplementedError("TODO: Implement TextProcessor.process")


class PDFProcessor:
    """
    Process PDF files - extract text content.

    Requires: PyPDF2 or pdfplumber library
    Install: pip install PyPDF2
    """

    SUPPORTED_MIMES = {"application/pdf"}

    def can_process(self, mime_type: str, filename: str) -> bool:
        """
        TODO: Check if mime_type is "application/pdf"
              OR if filename ends with ".pdf"
        """
        raise NotImplementedError("TODO: Implement can_process")

    def process(
        self, attachment: RawAttachment, openai_client: Any
    ) -> EnrichedAttachment:
        """
        Extract text from PDF.

        TODO: Implement this method
        =====
        Steps:
        1. Import: import base64, from io import BytesIO, from PyPDF2 import PdfReader
        2. Decode: pdf_bytes = base64.b64decode(attachment.data)
        3. Create reader: reader = PdfReader(BytesIO(pdf_bytes))
        4. Extract text from each page:
            pages = [page.extract_text() for page in reader.pages]
            content = "\\n\\n".join(pages)
        5. Build metadata: {"pages": len(reader.pages)}
        6. Truncate content if too large (e.g., MAX_CHARS = 15000)
        7. Return EnrichedAttachment with content and metadata

        Handle exceptions - return EnrichedAttachment with error set
        """
        raise NotImplementedError("TODO: Implement PDFProcessor.process")


class DocumentProcessor:
    """
    Process Word documents (docx, doc).

    Requires: python-docx library
    Install: pip install python-docx

    Note: .doc (old format) is harder to parse, may need different library
    """

    SUPPORTED_MIMES = {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
        "application/msword",  # .doc
    }

    def can_process(self, mime_type: str, filename: str) -> bool:
        """
        TODO: Check if mime_type is in SUPPORTED_MIMES
              OR if filename ends with ".docx" or ".doc"
        """
        raise NotImplementedError("TODO: Implement can_process")

    def process(
        self, attachment: RawAttachment, openai_client: Any
    ) -> EnrichedAttachment:
        """
        Extract text from Word document.

        TODO: Implement this method
        =====
        Steps:
        1. Import: import base64, from io import BytesIO, from docx import Document
        2. Decode: doc_bytes = base64.b64decode(attachment.data)
        3. Parse: doc = Document(BytesIO(doc_bytes))
        4. Extract: paragraphs = [p.text for p in doc.paragraphs]
                    content = "\\n".join(paragraphs)
        5. Truncate if too large
        6. Return EnrichedAttachment with content

        Note: This only works for .docx, not old .doc format
        Handle exceptions - return EnrichedAttachment with error set
        """
        raise NotImplementedError("TODO: Implement DocumentProcessor.process")


# ============================================================================
# MAIN PROCESSOR SERVICE
# ============================================================================


class AttachmentProcessor:
    """
    Main service for processing attachments.

    Usage:
        processor = AttachmentProcessor(openai_client)
        enriched = processor.process_attachments(raw_attachments)
        context_str = processor.build_llm_context(enriched)
    """

    def __init__(self, openai_client: Any = None):
        """
        Initialize with OpenAI client (needed for Vision API).

        TODO: Initialize self.openai_client
              Initialize self.processors list with processor instances
        """
        raise NotImplementedError("TODO: Implement __init__")

    def process_attachments(
        self, raw_attachments: List[Dict[str, Any]]
    ) -> List[EnrichedAttachment]:
        """
        Process all attachments and return enriched versions.

        TODO: Implement this method
        =====
        Args:
            raw_attachments: List of dicts from frontend payload
                Each dict has: type, filename, mime_type, data

        Steps:
        1. Loop through raw_attachments with enumerate for index
        2. Create RawAttachment from each dict
        3. Find appropriate processor using _find_processor
        4. Call processor.process() if found
        5. Use _create_fallback() if no processor or on error
        6. Collect and return list of EnrichedAttachment
        """
        raise NotImplementedError("TODO: Implement process_attachments")

    def _find_processor(self, attachment: RawAttachment) -> Optional[FileProcessor]:
        """
        Find the first processor that can handle this attachment.

        TODO: Loop through self.processors
              Return first one where can_process() returns True
              Return None if no processor found
        """
        raise NotImplementedError("TODO: Implement _find_processor")

    def _create_fallback(
        self, attachment: RawAttachment, error: str
    ) -> EnrichedAttachment:
        """
        Create a basic enriched attachment when processing fails.

        TODO: Return EnrichedAttachment with:
              - Basic info copied from attachment
              - attachment_type = AttachmentType.UNKNOWN
              - error message set
        """
        raise NotImplementedError("TODO: Implement _create_fallback")

    def build_llm_context(
        self,
        enriched_attachments: List[EnrichedAttachment],
        max_preview_chars: int = 500,
    ) -> str:
        """
        Build context string for LLM system/user prompt.

        TODO: Implement this method
        =====
        Output format:
            "User has uploaded 3 file(s):
            - chat_attachment_0 [IMAGE] chart.png: A bar chart...
            - chat_attachment_1 [PDF] report.pdf (5 pages): Summary...
            - chat_attachment_2 [TEXT] notes.txt: Meeting notes..."

        Steps:
        1. Return empty string if no attachments
        2. Build header line with count
        3. For each attachment, call to_llm_context(max_preview_chars)
        4. Join with newlines
        """
        raise NotImplementedError("TODO: Implement build_llm_context")

    def get_attachment_by_reference(
        self, reference: str, enriched_attachments: List[EnrichedAttachment]
    ) -> Optional[EnrichedAttachment]:
        """
        Look up attachment by reference string (e.g., "chat_attachment_0").

        TODO: Implement this method
        =====
        Used by tools (like send_email) to resolve attachment references.

        Steps:
        1. Check if reference starts with "chat_attachment_"
        2. Extract index: int(reference.split("_")[-1])
        3. Return enriched_attachments[index] if valid
        4. Return None on any error
        """
        raise NotImplementedError("TODO: Implement get_attachment_by_reference")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_attachment_processor(openai_client: Any = None) -> AttachmentProcessor:
    """
    Factory function to create AttachmentProcessor.

    TODO: Simply return AttachmentProcessor(openai_client)
    """
    raise NotImplementedError("TODO: Implement create_attachment_processor")
