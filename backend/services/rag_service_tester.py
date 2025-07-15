# from rag_service import RAGService
from backend.services.rag_service import RAGService
# from rag_service import extract_text_from_pdf
from pathlib import Path
from PyPDF2 import PdfReader # type: ignore

rag_service = RAGService()

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text content from uploaded PDF"""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

UPLOAD_DIR = Path("data/uploads")
PDF_DIR = UPLOAD_DIR / "pdf"

file_name="AI.pdf"

file_path = PDF_DIR / file_name
pdf_text =  extract_text_from_pdf(file_path)


# print(pdf_text)
from datetime import datetime
metadata = {
            "source_file": file_name,
            "title": file_name.replace('.pdf', ''),
            "upload_date": datetime.now().isoformat(),
            "file_path": str(file_path)
        }
async def f(file_name, pdf_text, metadata):
    success = await rag_service.add_document_to_index(
        document_id=file_name, content=pdf_text, metadata=metadata
    )
    return success

import asyncio
result = asyncio.run(f(file_name, pdf_text, metadata))
print(result)
