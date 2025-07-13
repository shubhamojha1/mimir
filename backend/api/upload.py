from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict
import os
from pathlib import Path

router = APIRouter(
    prefix="/upload",
    tags=["uploads"],
    responses={404: {"description": "Not found"}}
)

# Constants
UPLOAD_DIR = Path("data/uploads")
PDF_DIR = UPLOAD_DIR / "pdf"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/pdf", response_model=Dict[str, str])
async def upload_pdf(file: UploadFile = File(...)) -> JSONResponse:
    """
    Upload a PDF file to the server.
    
    Args:
        file (UploadFile): The PDF file to upload
        
    Returns:
        JSONResponse: Response containing filename and success message
        
    Raises:
        HTTPException: If file is not a PDF or if upload fails
    """
    try:
        # Validate file extension
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Only PDF files are allowed."
            )

        # Create safe filepath
        safe_filename = Path(file.filename).name  # Remove path traversal risk
        file_path = PDF_DIR / safe_filename

        # Save file
        async with open(file_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        return JSONResponse(
            content={
                "filename": safe_filename,
                "message": "PDF uploaded successfully"
            },
            status_code=200
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while uploading the file: {str(e)}"
        )