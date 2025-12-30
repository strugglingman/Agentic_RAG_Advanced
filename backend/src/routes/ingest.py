"""
Ingest routes blueprint.
Handles document ingestion endpoint.
"""

import os
import json
from flask import Blueprint, request, jsonify, g
from src.middleware.auth import require_identity
from src.utils.file_utils import get_upload_dir
from src.services.ingestion import ingest_one
from src.services.retrieval import build_bm25
from src.config.settings import Config

ingest_bp = Blueprint("ingest", __name__)

UPLOAD_BASE = Config.UPLOAD_BASE
FOLDER_SHARED = Config.FOLDER_SHARED


@ingest_bp.route("/ingest", methods=["POST"])
@require_identity
def ingest(vector_db):
    """
    Ingest documents into the vector database.
    Can ingest a single file or all files (when file_id="ALL").
    """
    body = request.get_json(force=True)
    file_id = body.get("file_id", "") if body else ""
    file_path = body.get("file_path", "") if body else ""
    file_path = (
        os.path.join(UPLOAD_BASE, file_path)
        if file_path and file_path != "ALL"
        else file_path
    )
    if not file_id:
        return jsonify({"message": "No correct file specified"}), 400
    if file_path != "ALL" and not os.path.exists(file_path):
        return jsonify({"message": "No correct file path specified"}), 400

    dept_id = g.identity.get("dept_id", "")
    if not dept_id:
        return jsonify({"error": "No organization ID provided"}), 400
    user_id = g.identity.get("user_id", "")
    if not user_id:
        return jsonify({"error": "No user ID provided"}), 400

    meta_data_all = []
    meta_data_files = []

    # Load metadata from user directory
    dir_user = get_upload_dir(base_path=UPLOAD_BASE, dept_id=dept_id, user_id=user_id)
    if dir_user:
        meta_data_files = [f for f in os.listdir(dir_user) if f.endswith(".meta.json")]
        for mf in meta_data_files:
            with open(os.path.join(dir_user, mf), "r", encoding="utf-8") as info_f:
                info = json.load(info_f)
                meta_data_all.append(info)

    # Load metadata from shared directory
    dir_shared = get_upload_dir(
        base_path=UPLOAD_BASE, dept_id=dept_id, user_id=FOLDER_SHARED
    )
    if dir_shared:
        meta_data_files = [
            f for f in os.listdir(dir_shared) if f.endswith(".meta.json")
        ]
        for mf in meta_data_files:
            with open(os.path.join(dir_shared, mf), "r", encoding="utf-8") as info_f:
                info = json.load(info_f)
                meta_data_all.append(info)

    ingested_info = ""
    if file_id == "ALL":
        for info in meta_data_all:
            fid = ingest_one(vector_db, info, app_user_id=user_id, app_dept_id=dept_id)
            if fid:
                ingested_info += f"{fid}\n"
    else:
        info = next((m for m in meta_data_all if m.get("file_id") == file_id), None)
        fid = ingest_one(vector_db, info, app_user_id=user_id, app_dept_id=dept_id)
        if fid:
            ingested_info = f"{fid}\n"

    # Rebuild BM25 index
    build_bm25(vector_db, dept_id, user_id)

    docs = vector_db.get(include=["documents"])["documents"]
    count = len(docs) if docs else 0
    ingested_info = f"{ingested_info}\n and the count of chunks is: {count}"

    msg = (
        f"Ingestion completed for file_ids:\n {ingested_info}"
        if ingested_info
        else "No new content ingested."
    )
    return jsonify({"message": msg}), 200
