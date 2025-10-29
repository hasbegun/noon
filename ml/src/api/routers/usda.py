"""
USDA food database endpoints
"""
from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import JSONResponse
from loguru import logger

from ..dependencies import get_usda_service
from ..models import USDASearchRequest

router = APIRouter(prefix="/api/v1/usda", tags=["usda"])


@router.post("/search")
async def search_usda_database(request: USDASearchRequest):
    """
    Search USDA food database

    Search for food items in the USDA FoodData Central database

    Args:
        request: Search parameters (query, limit, category)

    Returns:
        List of matching food items with FDC IDs and descriptions
    """
    usda_service = get_usda_service()
    if not usda_service:
        raise HTTPException(
            status_code=503,
            detail="USDA service not initialized"
        )

    try:
        logger.info(f"USDA search: '{request.query}' (limit={request.limit})")
        results = usda_service.search(
            query=request.query,
            limit=request.limit,
            category=request.category,
        )

        logger.info(f"USDA search returned {len(results)} results")
        return JSONResponse(content={"results": results})

    except Exception as e:
        logger.error(f"Error searching USDA database: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"USDA search failed: {str(e)}"
        )


@router.get("/food/{fdc_id}")
async def get_usda_food_by_id(
    fdc_id: int = Path(..., description="FDC ID of the food item"),
    portion_g: float = Query(100.0, gt=0, description="Portion size in grams"),
):
    """
    Get nutrition information for a specific food item

    Retrieve detailed nutrition facts for a food item by its FDC ID

    Args:
        fdc_id: FoodData Central ID
        portion_g: Portion size in grams (default: 100g)

    Returns:
        Nutrition information for the specified portion
    """
    usda_service = get_usda_service()
    if not usda_service:
        raise HTTPException(
            status_code=503,
            detail="USDA service not initialized"
        )

    try:
        logger.info(f"Fetching USDA food: FDC={fdc_id}, portion={portion_g}g")
        nutrition = usda_service.get_nutrition_for_portion(fdc_id, portion_g)

        if not nutrition:
            raise HTTPException(
                status_code=404,
                detail=f"Food item with FDC ID {fdc_id} not found"
            )

        return JSONResponse(content=nutrition)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving USDA food: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve food item: {str(e)}"
        )
