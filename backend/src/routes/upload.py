"""
Upload routes blueprint.
Handles file upload endpoint.
"""

import os
import json
import asyncio
from datetime import datetime
from flask import Blueprint, request, jsonify, g
from src.middleware.auth import require_identity
from src.utils.file_utils import validate_filename, create_upload_dir, canonical_path
from src.services.ingestion import make_id
from src.config.settings import Config
from src.services.file_manager import FileManager

upload_bp = Blueprint("upload", __name__)

UPLOAD_BASE = Config.UPLOAD_BASE
FOLDER_SHARED = Config.FOLDER_SHARED


@upload_bp.route("/upload", methods=["POST"])
@require_identity
def upload():
    """Upload a file and save metadata for ingestion."""
    if not request.files and not request.files.get("file"):
        return jsonify({"error": "No file part in the request"}), 400

    user_id = g.identity.get("user_id", "")
    dept_id = g.identity.get("dept_id", "")
    if not user_id or not dept_id:
        return jsonify({"error": "No user ID or organization ID provided"}), 400

    f = request.files["file"]
    filename = validate_filename(
        f,
        allowed_extensions=Config.ALLOWED_EXTENSIONS,
        mime_types=Config.MIME_TYPES,
    )
    if not filename:
        return jsonify({"error": "File is not valid mime type or extension"}), 400

    file_for_user = request.form.get("file_for_user", "0")
    upload_dir = create_upload_dir(
        base_path=UPLOAD_BASE, dept_id=dept_id, user_id=FOLDER_SHARED
    )
    if file_for_user == "1":
        upload_dir = create_upload_dir(
            base_path=UPLOAD_BASE, dept_id=dept_id, user_id=user_id
        )
    if not upload_dir:
        return jsonify({"error": "Failed to create upload directory"}), 500
    file_path = canonical_path(upload_dir, filename)

    # Check if same file exists
    if os.path.exists(file_path):
        return jsonify({"error": "File with the same name already exists"}), 400

    f.save(file_path)

    # Save file meta info for further ingestion
    tags = request.form.get("tags", "")
    tags_raw = json.loads(tags) if tags else []
    tags_str = ""
    if tags_raw:
        tags_str = ",".join(tags_raw) if tags_raw else ""

    file_size_bytes = os.path.getsize(str(file_path))
    file_info = {
        "file_id": make_id(filename),
        "file_path": str(file_path),
        "filename": filename,
        "source": filename,
        "ext": filename.rsplit(".", 1)[1].lower(),
        "size_kb": round(file_size_bytes / 1024, 1),
        "tags": tags_str,
        "upload_at": datetime.now().isoformat(),
        "uploaded_at_ts": datetime.now().timestamp(),
        "user_id": user_id,
        "dept_id": dept_id,
        "file_for_user": True if file_for_user == "1" else False,
        "ingested": False,
    }
    fileinfo_path = canonical_path(upload_dir, f"{filename}.meta.json")
    with open(fileinfo_path, "w", encoding="utf-8") as info_f:
        json.dump(file_info, info_f, indent=2)

    # Register file in FileRegistry for unified file management
    try:
        # Detect MIME type from extension
        ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""
        mime_type_map = {
            "pdf": "application/pdf",
            "doc": "application/msword",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "xls": "application/vnd.ms-excel",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "ppt": "application/vnd.ms-powerpoint",
            "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "txt": "text/plain",
            "csv": "text/csv",
            "md": "text/markdown",
            "json": "application/json",
            "xml": "application/xml",
        }
        mime_type = mime_type_map.get(ext, "application/octet-stream")

        async def register_uploaded_file():
            async with FileManager() as fm:
                return await fm.register_file(
                    user_email=user_id,
                    category="uploaded",
                    original_name=filename,
                    storage_path=str(file_path),
                    source_tool="upload_ui",
                    mime_type=mime_type,
                    size_bytes=file_size_bytes,
                    dept_id=dept_id,
                    indexed_in_chromadb=False,  # Will be set to True after ingestion
                    metadata=file_info,  # Store complete file_info from .meta.json
                )

        result = asyncio.run(register_uploaded_file())
        registry_file_id = result["file_id"]
        # Store registry_file_id in meta.json for cross-reference
        file_info["registry_file_id"] = registry_file_id
        with open(fileinfo_path, "w", encoding="utf-8") as info_f:
            json.dump(file_info, info_f, indent=2)
    except Exception as e:
        # Log error but don't fail the upload
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"[UPLOAD] Failed to register file in FileRegistry: {e}")

    return jsonify({"msg": "File uploaded successfully"}), 200
