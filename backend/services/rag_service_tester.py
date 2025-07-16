from backend.services.rag_service import RAGService
from backend.utils.prompts import RAGPrompts

from pathlib import Path
from PyPDF2 import PdfReader # type: ignore
import asyncio
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Command for testing: python -m backend.services.rag_service_tester

# rag_service = RAGService()

# def extract_text_from_pdf(file_path: str) -> str:
#     """Extract text content from uploaded PDF"""
#     with open(file_path, 'rb') as file:
#         reader = PdfReader(file)
#         text = ""
#         for page in reader.pages:
#             text += page.extract_text() or ""
#     return text

# UPLOAD_DIR = Path("data/uploads")
# PDF_DIR = UPLOAD_DIR / "pdf"

# file_name="AI.pdf"

# file_path = PDF_DIR / file_name
# pdf_text =  extract_text_from_pdf(file_path)


# # print(pdf_text)
# from datetime import datetime
# metadata = {
#             "source_file": file_name,
#             "title": file_name.replace('.pdf', ''),
#             "upload_date": datetime.now().isoformat(),
#             "file_path": str(file_path)
#         }
# async def test_indexing(file_name, pdf_text, metadata):
#     success = await rag_service.add_document_to_index(
#         document_id=file_name, content=pdf_text, metadata=metadata
#     )

#     logger.info(f"Indexing test result: {success}")
#     return success

# import asyncio
# result = asyncio.run(f(file_name, pdf_text, metadata))
# print(result)

class RAGServiceTester:
    def __init__(self):
        self.rag_service = RAGService()
        self.UPLOAD_DIR = Path("data/uploads")
        self.PDF_DIR = self.UPLOAD_DIR / "pdf"
        self.prompts = RAGPrompts()
        self.query = "What is Reinforcement Learning?"
        
    async def test_indexing(self):
        """Test document indexing"""
        file_name = "AI.pdf"
        file_path = self.PDF_DIR / file_name
        pdf_text = self.extract_text_from_pdf(file_path)
        
        metadata = {
            "source_file": file_name,
            "title": file_name.replace('.pdf', ''),
            "upload_date": datetime.now().isoformat(),
            "file_path": str(file_path)
        }

        success = await self.rag_service.add_document_to_index(
            document_id=file_name,
            content=pdf_text,
            metadata=metadata
        )
        logger.info(f"Indexing test result: {success}")
        return success
    
    async def test_retrieval(self):
        """Test context retrieval"""
        # query = "What are the key concepts of artificial intelligence?"
        # query = "What is Reinforcement Learning?"
        filters = {"source_file": "AI.pdf"}
        
        contexts = await self.rag_service.retrieve_relevant_context(
            query=self.query,
            filters=filters,
            max_results=3
        )
        
        logger.info(f"Retrieved {len(contexts)} contexts")
        for ctx in contexts:
            logger.info(f"Source: {ctx.source_document}")
            logger.info(f"Similarity: {ctx.similarity_score}")
            logger.info(f"Content preview: {ctx.content[:200]}...")
            logger.info("---")
        
        return contexts
    
    async def test_rag_response(self, contexts):
        """Test RAG response generation"""
        # query = "Summarize the main points about artificial intelligence"
        # query = "What is Reinforcement Learning?"
        
        response = await self.rag_service.generate_rag_response(
            query=self.query,
            contexts=contexts
        )
        
        logger.info(f"Answer: {response.answer}")
        logger.info(f"Confidence Score: {response.confidence_score}")
        logger.info(f"Processing Time: {response.processing_time}")
        return response

    def test_context_building(self, contexts):
        """Test context text building"""
        context_text = self.rag_service._build_context_text(contexts)
        logger.info(f"Built context text:\n{context_text}")
        return context_text

    async def test_llm_response(self, contexts):
        """Test LLM response generation using Ollama"""
        # system_prompt = "You are a helpful AI assistant."
        # # user_prompt = "Explain what is RAG in simple terms."
        # user_prompt = "Summarize the main points about artificial intelligence"
        context_text = self.test_context_building(contexts)

        system_prompt = self.prompts.get_rag_system_prompt("software development")
        user_prompt = self.prompts.get_rag_user_prompt(self.query, context_text)
        
        response = await self.rag_service._generate_llm_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        logger.info(f"LLM Response:\n{response}")
        return response

    def test_confidence_calculation(self, contexts):
        """Test confidence score calculation"""
        confidence = self.rag_service._calculate_confidence_score(contexts)
        logger.info(f"Calculated confidence score: {confidence}")
        return confidence

    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text content from uploaded PDF"""
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    

async def main():
    tester = RAGServiceTester()
    
    # Test indexing
    logger.info("Testing document indexing...")
    await tester.test_indexing()
    
    # Test retrieval
    logger.info("\nTesting context retrieval...")
    contexts = await tester.test_retrieval()
    
    # Test RAG response generation
    logger.info("\nTesting RAG response generation...")
    await tester.test_rag_response(contexts)
    
    # Test context building
    logger.info("\nTesting context building...")
    tester.test_context_building(contexts)
    
    # Test LLM response
    logger.info("\nTesting LLM response...")
    await tester.test_llm_response(contexts)
    
    # Test confidence calculation
    logger.info("\nTesting confidence calculation...")
    tester.test_confidence_calculation(contexts)

if __name__ == "__main__":
    asyncio.run(main())