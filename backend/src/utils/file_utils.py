"""File handling utilities"""

import os
import hashlib
from pathlib import Path
import magic
from werkzeug.utils import secure_filename


def make_id(text: str) -> str:
    """Generate MD5 hash ID from text"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def canonical_path(base: Path, *sub_paths: str) -> Path:
    """Resolve canonical path and prevent directory traversal"""
    base = base.resolve()
    upload_path = (base / Path(*sub_paths)).resolve()
    # check if upload_path is within base
    try:
        upload_path.relative_to(base)
    except ValueError:
        raise ValueError("Attempted directory traversal in upload path")
    return upload_path


def validate_filename(f, allowed_extensions: list, mime_types: list) -> str:
    """Validate uploaded file"""
    filename = secure_filename(f.filename)
    # Check file extension
    if not allowed_file(filename, allowed_extensions):
        return ""

    # Check mime type using python-magic-bin
    head = f.stream.read(8192)
    f.stream.seek(0)
    try:
        # python-magic-bin uses magic.from_buffer() directly
        mime = magic.from_buffer(head, mime=True) or ""
        mime = mime.lower()
    except Exception:
        # Fallback: if magic detection fails, just check extension
        mime = ""

    # If we got a mime type, verify it matches allowed types
    if mime:
        mime_ok = any(mime.startswith(x) for x in mime_types)
        if not mime_ok:
            return ""

    f.stream.seek(0)
    return filename


def allowed_file(filename: str, allowed_extensions: list) -> bool:
    """Check if file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def validate_filename_str(
    filename: str,
    allowed_extensions: list,
    content: bytes = None,
    mime_types: list = None,
) -> str:
    """
    Validate filename string (for FastAPI UploadFile).

    Args:
        filename: Original filename
        allowed_extensions: List of allowed extensions (e.g., ["pdf", "docx"])
        content: Optional file content bytes for MIME type validation
        mime_types: Optional list of allowed MIME types

    Returns:
        Sanitized filename if valid, empty string if invalid
    """
    # Sanitize filename
    sanitized = secure_filename(filename)
    if not sanitized:
        return ""

    # Check extension
    if not allowed_file(sanitized, allowed_extensions):
        return ""

    # Optional: Check MIME type from content
    if content and mime_types:
        try:
            head = content[:8192]
            mime = magic.from_buffer(head, mime=True) or ""
            mime = mime.lower()
            if mime:
                mime_ok = any(mime.startswith(x) for x in mime_types)
                if not mime_ok:
                    return ""
        except Exception:
            # If magic detection fails, just use extension check
            pass

    return sanitized


def create_upload_dir(
    base_path: str, dept_id: str, user_id: str, dept_split: str = "|"
) -> str:
    """Create upload directory for user"""
    try:
        folders = dept_id.split(dept_split)
        upload_dir = Path(base_path)
        upload_dir = canonical_path(upload_dir, *folders, user_id)
        upload_dir.mkdir(parents=True, exist_ok=True)
        return upload_dir
    except:
        return None


def get_upload_dir(
    base_path: str, dept_id: str, user_id: str, dept_split: str = "|"
) -> str:
    """Get upload directory path"""
    folders = dept_id.split(dept_split)
    try:
        upload_dir = Path(base_path)
        upload_dir = canonical_path(upload_dir, *folders, user_id)
        return str(upload_dir) if os.path.exists(str(upload_dir)) else ""
    except Exception:
        return ""
