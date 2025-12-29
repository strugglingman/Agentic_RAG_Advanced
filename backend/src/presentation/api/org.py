"""Organization API Router - FastAPI endpoint for organization structure."""

from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, status
from dishka.integrations.fastapi import FromDishka, inject
from src.application.queries.org import GetOrgStructureHandler


# ==================== RESPONSE MODELS ====================
class OrgStructureResponse(BaseModel):
    organization: str
    departments: list[dict]


# ==================== ROUTERS ====================
router = APIRouter(prefix="/org-structure", tags=["organization"])


# ==================== ENDPOINTS ====================
@router.get("", response_model=OrgStructureResponse)
@inject
async def get_org_structure(
    handler: FromDishka[GetOrgStructureHandler],
):
    result = await handler.execute()
    if result.error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error,
        )
    return OrgStructureResponse(**result.data)
