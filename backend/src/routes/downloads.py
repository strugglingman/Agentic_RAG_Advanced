"""
Downloads routes - Serves files downloaded by the download_file tool.

This module handles:
1. Serving downloaded files to users via /downloads/{user_id}/{filename}
2. Access control - users can only access their own files
3. Proper content-type headers for different file types
4. Security - prevents path traversal attacks

Architecture:
=============
    LLM downloads file → Saves to downloads/{user_id}/{filename} →
    Returns URL /downloads/{user_id}/{filename} →
    User clicks link → This route serves the file

TODO Implementation Steps:
==========================
Step 1: Setup route
    - Create Flask Blueprint: downloads_bp
    - Add route: @downloads_bp.get("/downloads/<user_id>/<filename>")
    - Apply @require_identity decorator for authentication

Step 2: Validate access permissions
    - Get current user_id from g.identity.get("user_id")
    - Compare with user_id from URL path
    - If mismatch: return 403 Forbidden error
    - This ensures users can only download their own files

Step 3: Validate and sanitize filename
    - Check for path traversal attempts: ".." in filename
    - Check for absolute paths: filename.startswith("/") or "\\"
    - If dangerous: return 400 Bad Request
    - Use os.path.basename() to ensure filename only (no directories)

Step 4: Build file path and check existence
    - Construct path: os.path.join(Config.DOWNLOAD_BASE, user_id, filename)
    - Use os.path.abspath() and verify it's within DOWNLOAD_BASE
    - Check if file exists: os.path.isfile(path)
    - If not found: return 404 Not Found

Step 5: Determine content type
    - Use mimetypes.guess_type(filename) to get MIME type
    - Fallback to "application/octet-stream" if unknown
    - Common types:
        - .pdf → application/pdf
        - .png → image/png
        - .xlsx → application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
        - .txt → text/plain

Step 6: Send file response
    - Use Flask's send_file() function
    - Set as_attachment=True to force download (optional)
    - Set download_name to original filename (for better UX)
    - Example: send_file(path, mimetype=content_type, as_attachment=True)

Step 7: Error handling
    - Wrap in try/except
    - Handle FileNotFoundError → 404
    - Handle PermissionError → 403
    - Handle other exceptions → 500 Internal Server Error
    - Log errors for debugging

Security Checklist:
===================
✓ Verify user owns the file (user_id matches)
✓ Prevent path traversal (no ../ or \\)
✓ Validate file is within DOWNLOAD_BASE directory
✓ Check file existence before serving
✓ Proper error messages (don't leak system paths)

Example Request Flow:
=====================
1. LLM returns: "👉 [report.pdf](/downloads/user123/20250118_120530_report.pdf)"
2. Frontend makes link clickable
3. User clicks → GET /downloads/user123/20250118_120530_report.pdf
4. This route:
   - Verifies user123 is current user
   - Validates filename
   - Checks downloads/user123/20250118_120530_report.pdf exists
   - Serves file with correct content-type
5. Browser downloads or displays file
"""

import os
import logging
import mimetypes
from flask import Blueprint, send_file, jsonify, g
from src.middleware.auth import require_identity
from src.config.settings import Config

logger = logging.getLogger(__name__)

downloads_bp = Blueprint("downloads", __name__)


@downloads_bp.get("/downloads/<user_id>/<filename>")
@require_identity
def download_file(user_id, filename):
    """
    Serve a downloaded file to the authenticated user.

    Args:
        user_id (str): User ID from URL path
        filename (str): Filename from URL path

    Returns:
        - File content with proper headers (on success)
        - JSON error message with status code (on failure)

    Security:
        - Only the file owner can download
        - Prevents path traversal attacks
        - Validates file exists within allowed directory
    """
    try:
        current_user_id = g.identity.get("user_id")
        if current_user_id != user_id:
            return (
                jsonify({"error": "Forbidden: You do not have access to this file."}),
                403,
            )

        # Sanitize filename
        if ".." in filename or filename.startswith(("/", "\\")):
            return jsonify({"error": "Invalid filename."}), 400

        safe_filename = os.path.basename(filename)
        download_dir = os.path.join(Config.DOWNLOAD_BASE, user_id)
        file_path = os.path.abspath(os.path.join(download_dir, safe_filename))
        download_abs_path = os.path.abspath(Config.DOWNLOAD_BASE)

        if not file_path.startswith(download_abs_path):
            return jsonify({"error": "Invalid file path."}), 400
        if not os.path.isfile(file_path):
            return jsonify({"error": "File not found."}), 404

        mime_type, _ = mimetypes.guess_type(safe_filename)
        if not mime_type:
            mime_type = "application/octet-stream"

        return send_file(
            file_path,
            mimetype=mime_type,
            as_attachment=True,
            download_name=safe_filename,
        )
    except FileNotFoundError:
        return jsonify({"error": "File not found."}), 404
    except PermissionError:
        return (
            jsonify(
                {"error": "Forbidden: You do not have permission to access this file."}
            ),
            403,
        )
    except Exception as e:
        logger.error(f"Error serving file {filename} for user {user_id}: {str(e)}")
        return jsonify({"error": "Internal server error."}), 500
