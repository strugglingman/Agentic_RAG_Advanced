"""
GetOrgStructure - Retrieve organization structure from JSON file.

No query parameters needed since it reads from a static config file.
"""

import json
from typing import Any, Optional
from dataclasses import dataclass
from src.application.common.interfaces import Query, QueryHandler
from src.config.settings import Config


@dataclass
class GetOrgStructureResult:
    data: dict[str, Any]
    error: Optional[str] = None


class GetOrgStructureHandler(QueryHandler[GetOrgStructureResult]):
    async def execute(self, query: Query = None) -> GetOrgStructureResult:
        try:
            with open(Config.ORG_STRUCTURE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

                return GetOrgStructureResult(data=data)
        except FileNotFoundError:
            return GetOrgStructureResult(
                data={}, error="Organization structure file not found"
            )
        except json.JSONDecodeError as e:
            return GetOrgStructureResult(data={}, error=f"Invalid JSON: {e}")
        except Exception as e:
            return GetOrgStructureResult(data={}, error=f"Unexpected error: {e}")
