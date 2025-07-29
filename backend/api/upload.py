from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Dict, Optional, Any
import os
from pathlib import Path
from PyPDF2 import PdfReader # type: ignore
from datetime import datetime
import logging

from ..services.rag_service import RAGService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler("app.log")  # Logs to a file
    ]
)

logger = logging.getLogger(__name__)

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

# initialize RAG service - chromaDB
rag_service = RAGService()

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text content from uploaded PDF"""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

@router.get("/ping")
async def check_health() -> JSONResponse:
    return JSONResponse(
        status_code=200,
        content={
            "message": "running!"
        }
    )

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
    print(f"Received file: {file}")
    print(f"Content type: {type(file)}")
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

        print(f"Received file: {file.filename}")
        print(f"Content type: {file.content_type}")

        # Save file
        print("======================File reading beigns======================")
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        print("======================File reading ends======================")

        print("======================Text extracting begins======================")
        pdf_text = extract_text_from_pdf(file_path)
        print("======================Text extracting ends======================")

        if not pdf_text.strip():
            return JSONResponse(
                content={
                    "filename": safe_filename,
                    "message": "PDF uploaded but no text could be extracted"
                },
                status_code=200
            )
        
        # creating metadata for the document
        metadata = {
            "source_file": safe_filename,
            "title": safe_filename.replace('.pdf', ''),
            "upload_date": datetime.now().isoformat(),
            "file_path": str(file_path)
        }
        print("======================indexing begins======================")
        success = await rag_service.add_document_to_index(
            document_id = safe_filename,
            content=pdf_text,
            metadata=metadata
        )
        print("======================indexing ends======================")

        if success:
            return JSONResponse(
                content={
                    "filename": safe_filename,
                    "message": "PDF uploaded and indexed successfully"
                },
                status_code=200
            )
        else:
            return JSONResponse(
                content={
                    "filename": safe_filename,
                    "message": "PDF uploaded but indexing failed"
                },
                status_code=500
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while uploading the file: {str(e)}"
        )

from pydantic import BaseModel
from typing import Dict, Any, Optional
# TODO: Separate endpoints based on logic
class SearchRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None

@router.post("/rag")
async def semantic_search(
    # query: str = Query(..., description="The search query"),
    # filters: Optional[Dict[str, Any]] = None,
    request: SearchRequest
) -> Dict[str, Any]:
    """
    Perform semantic search on uploaded documents and return RAG-powered response.
    """
    try:
        # retrieve relevant contexts
        logger.info("=========Retrieve relevant context start=========")
        contexts = await rag_service.retrieve_relevant_context(
            query=request.query,
            filters=request.filters,
            max_results=3 # TODO: Need to adjust accordingly.
        )
        logger.info(f"Retrieved {len(contexts)} contexts")
        logger.info("=========Retrieve relevant context end=========")

        logger.info("=========Generate RAG response start=========")
        # generate RAG Response using contexts
        response = await rag_service.generate_rag_response(
            query=request.query,
            contexts=contexts,
            agent_type="software development" # AGENT TYPE MAKES ALL THE DIFFERENCE BETWEEN A GOOD AND BAD RESPONSE.
                                              # TODO: Need to address this so that agent type is inferred dynamically.
        )
        logger.info("=========Generate RAG response end=========")

        return {
            "answer": response.answer,
            "sources": [
                {
                    "content": source.content,
                    "document": source.source_document,
                    "similarity": source.similarity_score,
                    "metadata": source.metadata
                }
                for source in response.sources
            ],
            "confidence_score": response.confidence_score,
            "processing_time": response.processing_time
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error performing semantic search: {str(e)}"
        )
