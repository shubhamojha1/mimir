import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
from datetime import datetime

import chromadb
from chromadb.config import Settings as ChromaSettings
# import openai
# from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
import tiktoken
import os
import time
print(os.getcwd())

from ..config.settings import get_settings
from backend.utils.prompts import RAGPrompts

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
        self.settings = get_settings()
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2') # best free alternative to openai embeddings
        self.prompts = RAGPrompts()

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
            start_time = datetime.now()
            chunks = self._create_chunks(content)

            embeddings = []
            chunk_ids = []
            chunk_metadata = []

            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                chunk_embedding = self.embedding_model.encode(chunk).tolist()
                
                # Combine document metadata with chunk-specific information
                chunk_meta = {
                    **metadata,
                    "chunk_index": i,
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "content_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
                }
                
                embeddings.append(chunk_embedding)
                chunk_ids.append(chunk_id)
                chunk_metadata.append(chunk_meta)
            
            self.collection.add(
                embeddings = embeddings,
                documents = chunks,
                metadatas = chunk_metadata,
                ids = chunk_ids
            )
            processing_time = (datetime.now() - start_time).total_seconds()

            logger.info(f"Added document {document_id} with {len(chunks)} chunks to RAG index. Took {processing_time} seconds.")
            return True
        except Exception as e:
            logger.error(f"Failed to add document {document_id} to RAG index: {e}")
            return False
        
    def _create_chunks(self, content: str) -> List[str]:
        """
        Split content into overlapping chunks for better retrieval.
        
        Chunking is crucial for RAG because it allows us to find specific relevant
        sections within large documents. The overlap ensures that important information
        spanning chunk boundaries isn't lost.
        """
        chunk_size = self.settings.rag_chunk_size
        overlap = self.settings.rag_chunk_overlap

        chunks = []
        start = 0

        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]

            # if not at end, try to break at a word boundary
            if end < len(content):
                last_space = chunk.rfind(' ')
                if last_space > chunk_size * 0.8: # break only if we dont lose too much content
                    chunk = chunk[:last_space]
                    end = start + last_space

            chunks.append(chunk.strip())
            start = end - overlap

            # prevent infinite loop
            if start >= len(content):
                break

        return [chunk for chunk in chunks if chunk.strip()]
    
    async def retrieve_relevant_context(self, 
                                        query: str, 
                                        filters: Optional[Dict[ str, Any]] = None,
                                        max_results: Optional[int] = None) -> List[RAGContext]:
        """
        Retrieve the most relevant document chunks for a given query.
        
        This is the "Retrieval" part of RAG. We convert the query to a vector embedding
        and find the most semantically similar chunks in our database.
        
        Args:
            query: The user's question or search query
            filters: Optional filters (e.g., subject, document type, date range)
            max_results: Maximum number of results to return
        
        Returns:
            List of RAGContext objects containing relevant document chunks
        """
        try:
            # generate embedding for the query
            query_embedding = self.embedding_model.encode(query).tolist()

            # determine number of results to review
            n_results = max_results or self.settings.rag_max_results

            # build ChromaDB query params
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": n_results
            }

            # add filters if provided
            if filters:
                query_params["where"] = filters

            # query ChromaDB for similar chunks
            results = self.collection.query(**query_params)

            # convert results to RAGContext objects
            contexts = []
            for i in range(len(results["documents"][0])):
                context = RAGContext(
                    content=results["documents"][0][i],
                    source_document=results["metadatas"][0][i].get("source_file", "Unknown"),
                    chunk_id=results["ids"][0][i],
                    similarity_score=1.0 - results["distances"][0][i],  # Convert distance to similarity
                    metadata=results["metadatas"][0][i],
                    timestamp=datetime.now()
                )

                # only include results above similarity threshold
                if context.similarity_score >= self.settings.rag_similarity_threshold:
                    contexts.append(context)

            logger.info(f"Retrieved {len(contexts)} relevant contexts for query: {query[:50]}...")
            return contexts
        
        except Exception as e:
            logger.error(f"Failed to retrieve relevant context: {e}")
            return []
        
    async def generate_rag_response(self,
                                    query: str,
                                    contexts: List[RAGContext],
                                    agent_type: Optional[str] = None) -> RAGResponse:
        """
        Generate a response using retrieved context and LLM.
        
        This is the "Generation" part of RAG. We combine the user's query with
        relevant context from their documents and ask the LLM to generate a
        comprehensive, grounded answer.
        
        Args:
            query: The user's question
            contexts: Relevant document chunks retrieved earlier
            agent_type: Optional agent specialization (math, science, etc.)
        
        Returns:
            RAGResponse containing the answer and source information
        """
        start_time = datetime.now()
        try:
            # build context string from retrieved chunks
            context_text = self._build_context_text(contexts)

            # select appropriate prompt based on agent type
            system_prompt = self.prompts.get_rag_system_prompt(agent_type)
            user_prompt = self.prompts.get_rag_user_prompt(query, context_text)

            # generate response using configured LLM
            answer = await self._generate_llm_response(system_prompt, user_prompt)

            # calculate confidence score based on context relevance
            confidence_score = self._calculate_confidence_score(contexts)

            processing_time = (datetime.now() - start_time).total_seconds()

            response = RAGResponse(
                answer=answer,
                sources=contexts,
                query=query,
                confidence_score=confidence_score,
                processing_time=processing_time
            )

            logger.info(f"Generated RAG response in {processing_time:.2f}s with confidence {confidence_score:.2f}")
            return response

        except Exception as e:
            logger.error(f"Failed to generate RAG response: {e}")
            return RAGResponse(
                answer="I'm sorry, but I encountered an error while processing your request. Please try again.",
                sources=[],
                query=query,
                confidence_score=0.0,
                processing_time=(datetime.now() - start_time).total_seconds()                
            )
    
    def _build_context_text(self, contexts: List[RAGContext]) -> str:
        """
        Combine retrieved contexts into a single text block for the LLM.
        
        This step is crucial because it determines how much context the LLM sees
        and how it's formatted. We include source information to maintain transparency.
        """
        if not contexts:
            return "No relevant context found in your documents."
        
        context_parts = []
        for i, context in enumerate(contexts, 1):
            context_part = f"[Source {i}: {context.source_document}]\n{context.content}\n"
            context_parts.append(context_part)

        return "\n".join(context_parts)
    
    async def _generate_llm_response(self, system_prompt, user_prompt) -> str:
        """
        Generate response using the configured LLM provider.
        
        This method abstracts away the differences between various LLM providers,
        allowing Mimir to work with OpenAI, Anthropic, or local models seamlessly.
        """
        if self.settings.llm_provider == "openai":
            return await self._generate_openai_response(system_prompt, user_prompt)
        elif self.settings.llm_provider == "anthropic":
            return await self._generate_anthropic_response(system_prompt, user_prompt)
        elif self.settings.llm_provider == "ollama":
            return await self._generate_ollama_response(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.settings.llm_provider}")
        
    async def _generate_openai_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using OpenAI's API."""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def _generate_anthropic_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using Anthropic's API."""
        try:
            response = await self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    async def _generate_ollama_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using local Ollama model."""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": "llama3",
                    "prompt": f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:",
                    "stream": False
                }
                
                async with session.post(f"{self.ollama_host}/api/generate", json=payload) as response:
                    result = await response.json()
                    return result.get("response", "")
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise

    def _calculate_confidence_score(self, contexts: List[RAGContext]) -> float:
        """
        Calculate confidence score based on context relevance.
        
        This helps users understand how confident the system is in its answer
        based on the quality and relevance of the retrieved context.
        """
        if not contexts:
            return 0.0

        # avg similarity score of all contexts
        avg_similarity = sum(ctx.similarity_score for ctx in contexts) / len(contexts)

        # bonus for having multiple relevant sources
        source_bonus = min(len(contexts) / 3, 1.0) * 0.1

        return min(avg_similarity + source_bonus, 1.0)

    async def search_documents(self,
                               query,
                               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search documents and return formatted results.
        
        This provides a more traditional search experience while still leveraging
        the semantic search capabilities of the RAG system.
        """
        contexts = await self.retrieve_relevant_context(query, filters)

        # Group contexts by document and format results
        document_results = {}
        for context in contexts:
            doc_id = context.metadata.get("document_id")
            if doc_id not in document_results:
                document_results[doc_id] = {
                    "document_id": doc_id,
                    "title": context.metadata.get("title", "Untitled"),
                    "source_file": context.source_document,
                    "subject": context.metadata.get("subject", "General"),
                    "upload_date": context.metadata.get("upload_date"),
                    "relevant_chunks": [],
                    "max_similarity": 0.0
                }
            
            document_results[doc_id]["relevant_chunks"].append({
                "content": context.content[:200] + "..." if len(context.content) > 200 else context.content,
                "similarity_score": context.similarity_score
            })

            # Track highest similarity score for ranking
            document_results[doc_id]["max_similarity"] = max(
                document_results[doc_id]["max_similarity"],
                context.similarity_score
            )

        # Sort by relevance
        sorted_results = sorted(
            document_results.values(),
            key=lambda x: x["max_similarity"],
            reverse=True
        )

        return sorted_results


