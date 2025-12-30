"""Organization API Router - FastAPI endpoint for organization structure."""

from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, status, Request
from dishka.integrations.fastapi import FromDishka, inject
from slowapi import Limiter
from slowapi.util import get_remote_address
from src.application.queries.org import GetOrgStructureHandler

# Rate limiter for this router
limiter = Limiter(key_func=get_remote_address)


# ==================== RESPONSE MODELS ====================
class OrgStructureResponse(BaseModel):
    organization: str
    departments: list[dict]


# ==================== ROUTERS ====================
router = APIRouter(prefix="/org-structure", tags=["organization"])


# ==================== ENDPOINTS ====================
@router.get("", response_model=OrgStructureResponse)
@limiter.limit("1/minute;10/day")
@inject
async def get_org_structure(
    request: Request,
    handler: FromDishka[GetOrgStructureHandler],
):
    """
    Get organization structure.

    Rate limited: 1/minute, 10/day (matches Flask - stricter limit)
    """
    result = await handler.execute()
    if result.error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error,
        )
    return OrgStructureResponse(**result.data)
