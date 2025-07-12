import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
from datetime import datetime

import chromadb
from chromadb.config import Settings as ChromaSettings
import openai
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
import tiktoken

# from backend.config.settings import get_settings
# from backend.utils.prompts import RAGPrompts

logger = logging.getLogger(__name__)

# CHROMA_DIR = "data/chroma_db"
# os.makedirs(CHROMA_DIR, exist_ok=True)

# client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=True))
# collection = client.get_or_create_collection("pdfs")

# def extract_text_from_pdf(pdf_path):
#     reader = PdfReader(pdf_path)
#     text = ""
#     for page in reader.pages:
#         text += page.extract_text() or ""
#     return text

@dataclass
class RAGContext:
    """
    Represents a piece of retrieved context with metadata.
    This helps us track where information came from and how relevant it is.
    """
    content: str
    source_document: str
    chunk_id: str
    similarity_score: float
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class RAGResponse:
    """
    Complete response from the RAG system including the answer and sources.
    This transparency helps users understand how the AI reached its conclusions.
    """
    answer: str
    sources: List[RAGContext]
    query: str
    confidence_score: float
    processing_time: float

class RAGService:
    """
    The RAG Service orchestrates the entire retrieval-augmented generation process.
    
    This service acts as the bridge between your documents and AI responses, ensuring
    that every answer is grounded in your actual uploaded content rather than relying
    solely on pre-trained knowledge.
    """

    def __init__(self):
        # self.settings = get_settings()
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2') # best free alternative to openai embeddings
        # self.prompts = RAGPrompts()

        self._init_vector_db() # initialize chromadb
        self._init_llm_clients() # initialize llm clients based on configurations

        # token counter for managing context windows
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo") # need to check which model

        logger.info("RAG Service initialized successfully!")

    def _init_vector_db(self):
        """
        Initialize the vector database for semantic search.
        
        ChromaDB stores document embeddings (vector representations) that allow us
        to find semantically similar content even when the exact words don't match.
        """
        try:
            # Create ChromaDB client with persistent storage
            self.chroma_client = chromadb.PersistentClient(
                path="./data/chroma_db",
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            # Get or create collection for documents
            self.collection = self.chroma_client.get_or_create_collection(
                name='pdfs', #self.settings.vector_collection_name,
                metadata={"description": "Mimir document embeddings"}
            )
            
            logger.info(f"Vector database initialized with {self.collection.count()} documents")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            raise

    def _init_llm_clients(self):
        """
        Initialize the appropriate LLM client based on configuration.
        
        This flexibility allows Mimir to work with different AI providers or
        even local models, making it adaptable to various deployment scenarios.
        """
        if self.settings.llm_provider == "openai" and self.settings.openai_api_key:
            self.openai_client = openai.OpenAI(api_key=self.settings.openai_api_key)
            logger.info("OpenAI client initialized")
            
        elif self.settings.llm_provider == "anthropic" and self.settings.anthropic_api_key:
            self.anthropic_client = Anthropic(api_key=self.settings.anthropic_api_key)
            logger.info("Anthropic client initialized")
            
        elif self.settings.llm_provider == "ollama":
            # For local Ollama models, we'll use HTTP requests
            self.ollama_host = self.settings.ollama_host
            logger.info("Ollama client configured")
            
        else:
            raise ValueError("No valid LLM provider configured")
        
    async def add_document_to_index(self, document_id, content, metadata) -> bool:
        """
        Add a document to the RAG index for future retrieval.
        
        This process breaks down documents into smaller chunks (typically 1000 characters)
        with some overlap to ensure important information isn't lost at chunk boundaries.
        Each chunk gets converted to a vector embedding that captures its semantic meaning.
        
        Args:
            document_id: Unique identifier for the document
            content: Full text content of the document
            metadata: Additional information (title, subject, upload date, etc.)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            chunks = self._create_chunks(content)

            embeddings = []
            chunk_ids = []
            chunk_metadata = []

            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                chunk_embedding = self.embedding_model.encode(chunk).tolist()

                chunk_metadata = {
                    **metadata,
                    "chunk_index": i,
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "content_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
                }

                embeddings.append(chunk_embedding)
                chunk_ids.append(chunk_id)
                chunk_metadata.append(chunk_metadata)
            
            self.collection.add(
                embeddings = embeddings,
                documents = chunks,
                metadatas = chunk_metadata,
                ids = chunk_ids
            )

            logger.info(f"Added document {document_id} with {len(chunks)} chunks to RAG index")
            return True
        except Exception as e:
            logger.error(f"Failed to add document {document_id} to RAG index: {e}")
            return False