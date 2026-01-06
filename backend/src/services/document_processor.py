"""
Document processing service.

Supports multilingual document processing including:
- Space-separated languages: English, Swedish, Finnish, Spanish, German, French
- CJK languages: Chinese, Japanese (proper sentence splitting)
"""
import os
import re
import csv
import json
import logging
from docx import Document
from src.utils.multilingual import split_sentences as multilingual_split_sentences

logger = logging.getLogger(__name__)


def _clean_pdf_text(text: str) -> str:
    """Clean extracted PDF text by removing artifacts and normalizing whitespace."""
    if not text:
        return ""
    # Remove multiple spaces
    text = re.sub(r" {2,}", " ", text)
    # Remove multiple newlines (keep max 2)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove hyphenation at line breaks (e.g., "docu-\nment" -> "document")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


def _extract_with_pymupdf(file_path: str) -> list:
    """Extract text using PyMuPDF (fitz) - fast and reliable."""
    try:
        import fitz  # pymupdf
        doc = fitz.open(file_path)
        pages_text = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            text = _clean_pdf_text(text)
            if text:
                pages_text.append((page_num, text))
        doc.close()
        if pages_text:
            logger.debug(f"[PDF] PyMuPDF extracted {len(pages_text)} pages from {file_path}")
        return pages_text
    except Exception as e:
        logger.debug(f"[PDF] PyMuPDF failed for {file_path}: {e}")
        return []


def _extract_with_pdfplumber(file_path: str) -> list:
    """
    Extract text using pdfplumber - better for tables and complex layouts.

    Strategy: Extract text outside table areas, then append formatted tables.
    This avoids duplicating table content.
    """
    try:
        import pdfplumber
        pages_text = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Find all tables on the page
                tables = page.find_tables()

                if tables:
                    # Get bounding boxes of all tables
                    table_bboxes = [table.bbox for table in tables]

                    # Filter out characters that are inside table areas
                    def not_in_table(obj):
                        """Check if object is outside all table bounding boxes."""
                        obj_x = (obj["x0"] + obj["x1"]) / 2
                        obj_y = (obj["top"] + obj["bottom"]) / 2
                        for bbox in table_bboxes:
                            x0, top, x1, bottom = bbox
                            if x0 <= obj_x <= x1 and top <= obj_y <= bottom:
                                return False
                        return True

                    # Extract text excluding table areas
                    filtered_page = page.filter(not_in_table)
                    text = filtered_page.extract_text() or ""

                    # Extract tables as formatted text
                    table_texts = []
                    for table in tables:
                        extracted = table.extract()
                        if extracted:
                            rows = []
                            for row in extracted:
                                if row:
                                    row_text = " | ".join(
                                        str(cell).strip() if cell else "" for cell in row
                                    )
                                    rows.append(row_text)
                            if rows:
                                table_texts.append("\n".join(rows))

                    # Combine text and tables
                    if table_texts:
                        text = text + "\n\n[TABLE]\n" + "\n\n[TABLE]\n".join(table_texts)
                else:
                    # No tables - just extract text normally
                    text = page.extract_text() or ""

                text = _clean_pdf_text(text)
                if text:
                    pages_text.append((page_num, text))

        if pages_text:
            logger.debug(f"[PDF] pdfplumber extracted {len(pages_text)} pages from {file_path}")
        return pages_text
    except Exception as e:
        logger.debug(f"[PDF] pdfplumber failed for {file_path}: {e}")
        return []


def _extract_with_pypdf(file_path: str) -> list:
    """Extract text using pypdf - fallback option."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        pages_text = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = _clean_pdf_text(text)
            if text:
                pages_text.append((page_num, text))
        if pages_text:
            logger.debug(f"[PDF] pypdf extracted {len(pages_text)} pages from {file_path}")
        return pages_text
    except Exception as e:
        logger.debug(f"[PDF] pypdf failed for {file_path}: {e}")
        return []


def _extract_pdf_with_fallback(file_path: str) -> list:
    """
    Extract text from PDF using fallback chain:
    1. PyMuPDF (fitz) - fastest, good general extraction
    2. pdfplumber - better for tables and complex layouts
    3. pypdf - fallback

    Returns the result from the first method that extracts meaningful content.
    """
    # Try PyMuPDF first (fastest)
    result = _extract_with_pymupdf(file_path)
    if result:
        return result

    # Try pdfplumber (better for tables)
    result = _extract_with_pdfplumber(file_path)
    if result:
        return result

    # Fallback to pypdf
    result = _extract_with_pypdf(file_path)
    if result:
        return result

    logger.warning(f"[PDF] All extraction methods failed for {file_path}")
    return []


def read_text(file_path: str, text_max: int = 400000):
    """Read text from various file formats"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return _extract_pdf_with_fallback(file_path)
    
    if ext == ".csv":
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            all_rows = [",".join(row) for row in reader]
            return [(0, "\n".join(all_rows)[:text_max])]
    
    if ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return [(0, json.dumps(data, indent=2)[:text_max])]
            except Exception:
                return [(0, f.read()[:text_max])]
    
    if ext == ".docx":
        doc = Document(file_path)
        # Iterate through document body in order (preserves paragraph/table sequence)
        text_parts = []
        for element in doc.element.body:
            # Check if element is a paragraph
            if element.tag.endswith("p"):
                # Find corresponding paragraph object
                for para in doc.paragraphs:
                    if para._element is element and para.text.strip():
                        text_parts.append(para.text)
                        break
            # Check if element is a table
            elif element.tag.endswith("tbl"):
                # Find corresponding table object
                for table in doc.tables:
                    if table._element is element:
                        table_rows = []
                        for row in table.rows:
                            row_text = " | ".join(cell.text.strip() for cell in row.cells)
                            if row_text.strip():
                                table_rows.append(row_text)
                        if table_rows:
                            text_parts.append("[TABLE]\n" + "\n".join(table_rows))
                        break
        return [(0, "\n\n".join(text_parts)[:text_max])]
    
    with open(file_path, "r", encoding="utf-8") as f:
        return [(0, f.read()[:text_max])]


def sentence_split(text: str) -> list[str]:
    """
    Split text into sentences with multilingual support.

    Handles:
    - Latin languages: . ! ? followed by space and capital letter
    - Chinese: 。 ！ ？ ；
    - Japanese: 。 ！ ？
    - Mixed multilingual text
    """
    return multilingual_split_sentences(text)


def make_chunks(pages_text: list, target: int = 400, overlap: int = 90) -> list[tuple]:
    """Split document into overlapping chunks"""
    all_chunks = []
    
    for page_num, text in pages_text:
        chunks, buff, size = [], [], 0
        sentences = sentence_split(text)
        
        for s in sentences:
            buff.append(s)
            if size + len(s) <= target:
                size += len(s) + 1
            else:
                if buff:
                    chunks.append((page_num, ' '.join(buff)))
                
                overlap_sentences = []
                overlap_size = 0
                for sent in reversed(buff):
                    if overlap_size + len(sent) + (1 if overlap_sentences else 0) <= overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_size += len(sent) + (1 if overlap_size > 0 else 0)
                    else:
                        break
                
                buff = overlap_sentences
                size = sum(len(s) for s in buff) + max(0, len(buff) - 1)
        
        if buff:
            chunks.append((page_num, ' '.join(buff)))
        
        all_chunks.extend(chunks)
    
    return all_chunks
