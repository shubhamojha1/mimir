from fastapi import APIRouter, UploadFile, File, HTTPException
import os

router = APIRouter()
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    PDF_FILE_DIR = UPLOAD_DIR+"/pdf"
    os.makedirs(PDF_FILE_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR+"/pdf", file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"filename": file.filename, "message": "PDF uploaded successfully."}
