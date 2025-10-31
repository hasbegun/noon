# app/api/analyze_image.py

from fastapi import UploadFile, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from service import inference_service
from storage_service import save_image_and_analysis, SessionLocal

import logging
logger = logging.getLogger('app.api.analyze_image')

async def handle_image_analysis(
    background_tasks: BackgroundTasks,
    image: UploadFile,
) -> dict:
    """
    Handles the business logic for the image analysis endpoint.
    1. Validates the image file.
    2. Calls the inference service to get the analysis.
    3. Dispatches a background task to save the image and analysis results.
    4. Returns the analysis result.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image.")

    image_bytes = await image.read()
    original_filename = image.filename or "upload.jpg"

    # A database session is now required for the main analysis logic
    db = SessionLocal()
    try:
        # Call the inference service, which now has its own default logic.
        # analysis_result = await inference_service.analyze_image(image_bytes)
        analysis_result = await inference_service.analyze_image_with_db(db, image_bytes)

        # Dispatch the background task to save the results.
        background_tasks.add_task(
            save_image_and_analysis,
            image_bytes=image_bytes,
            original_filename=original_filename,
            analysis_data=analysis_result
        )

    except HTTPException:
        # Re-raise HTTP exceptions from the service layer
        raise
    except Exception as e:
        logger.error("Error during LLM analysis: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error during analysis.")
    finally:
        # It's crucial to close the session to release the connection
        db.close()

    # Step 3: Return the analysis result to the router.
    return {"analysis": analysis_result}
