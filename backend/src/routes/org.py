"""
Organization structure routes blueprint.
Handles organization structure endpoint.
"""

import json
from flask import Blueprint, jsonify
from src.config.settings import Config

org_bp = Blueprint("org", __name__)

ORG_STRUCTURE_FILE = Config.ORG_STRUCTURE_FILE


@org_bp.route("/org-structure", methods=["GET"])
def org_structure():
    """Get organization structure from JSON file."""
    try:
        with open(ORG_STRUCTURE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return jsonify(data), 200
    except Exception as ex:
        return (
            jsonify({"error": f"Failed to read organization structure: {str(ex)}"}),
            500,
        )
