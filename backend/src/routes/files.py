"""
Files routes blueprint.
Handles file listing and download endpoints.
"""

import os
import json
import asyncio
from flask import Blueprint, request, jsonify, g, send_file
from src.middleware.auth import require_identity
from src.utils.file_utils import get_upload_dir
from src.config.settings import Config
from src.services.file_manager import FileManager

files_bp = Blueprint("files", __name__)

UPLOAD_BASE = Config.UPLOAD_BASE
FOLDER_SHARED = Config.FOLDER_SHARED


@files_bp.route("/files", methods=["GET"])
@require_identity
def list_files():
    """List all files accessible to the current user."""
    dept_id = g.identity.get("dept_id", "")
    if not dept_id:
        return jsonify({"error": "No organization ID provided"}), 400
    user_id = g.identity.get("user_id", "")
    if not user_id:
        return jsonify({"error": "No user ID provided"}), 400

    files_info = []

    # List this user's files first
    dir_user = get_upload_dir(base_path=UPLOAD_BASE, dept_id=dept_id, user_id=user_id)
    if dir_user:
        files = os.listdir(dir_user)
        for f in files:
            if f.endswith("meta.json"):
                with open(os.path.join(dir_user, f), "r", encoding="utf-8") as info_f:
                    info = json.load(info_f)
                    if info.get("dept_id", "") == dept_id and (
                        (not info.get("file_for_user", False))
                        or info.get("user_id") == user_id
                    ):
                        # Include a sanitized or relative file path
                        info["file_path"] = os.path.relpath(
                            info.get("file_path", ""), UPLOAD_BASE
                        )
                        files_info.append(info)

    # List shared files next
    dir_shared = get_upload_dir(
        base_path=UPLOAD_BASE, dept_id=dept_id, user_id=FOLDER_SHARED
    )
    if dir_shared:
        files = os.listdir(dir_shared)
        for f in files:
            if f.endswith("meta.json"):
                with open(os.path.join(dir_shared, f), "r", encoding="utf-8") as info_f:
                    info = json.load(info_f)
                    if info.get("dept_id", "") == dept_id and (
                        (not info.get("file_for_user", False))
                        or info.get("user_id") == user_id
                    ):
                        # Include a sanitized or relative file path
                        info["file_path"] = os.path.relpath(
                            info.get("file_path", ""), UPLOAD_BASE
                        )
                        files_info.append(info)

    return jsonify({"files": files_info}), 200


@files_bp.route("/files/<file_id>", methods=["GET"])
@require_identity
def download_file(file_id):
    """
    Download a file by its FileRegistry ID.
    Serves files from any location (uploads, downloads, chat) securely.
    """
    user_email = g.identity.get("user_id", "")
    if not user_email:
        return jsonify({"error": "No user ID provided"}), 400

    try:
        # Get file path from FileManager (includes security check)
        async def get_file():
            async with FileManager() as fm:
                return await fm.get_file_path(file_id, user_email)

        file_path = asyncio.run(get_file())

        # Verify file exists
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found on disk"}), 404

        # Serve the file
        return send_file(
            file_path,
            as_attachment=True,
            download_name=os.path.basename(file_path)
        )

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except PermissionError as e:
        return jsonify({"error": "Access denied"}), 403
    except Exception as e:
        return jsonify({"error": f"Failed to download file: {str(e)}"}), 500
