from ..services.agent_service import AgentService
from ..services.rag_service import RAGService

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime, timezone
import uuid
import httpx
import logging

rag_service = RAGService()
agent_service = AgentService(rag_service=rag_service)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={404: {"description": "Not found"}}
)

def get_utc_now() -> datetime:
    """Helper function to get current UTC time with timezone info"""
    return datetime.now(timezone.utc)

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=get_utc_now)

class ChatSession(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    created_at: datetime = Field(default_factory=get_utc_now)
    last_updated: datetime = Field(default_factory=get_utc_now)
    metadata: Dict = Field(default_factory=dict)

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    prompt: str
    model: str = "llama2"  # Changed default model
    context: Optional[Dict] = None

# Initialize session storage
chat_sessions: Dict[str, ChatSession] = {}

def create_chat_session() -> ChatSession:
    """Create a new chat session with UUID"""
    session_id = str(uuid.uuid4())
    session = ChatSession(
        session_id=session_id,
        messages=[]
    )
    chat_sessions[session_id] = session
    logger.info(f"Created new chat session: {session_id}")
    return session

async def validate_ollama_connection(client: httpx.AsyncClient, model: str) -> bool:
    """Validate Ollama server connection and model availability"""
    try:
        # Check if Ollama is running
        response = await client.get("http://localhost:11434/api/version")
        response.raise_for_status()
        
        # Check if model exists
        model_response = await client.post(
            "http://localhost:11434/api/show", 
            json={"name": model}
        )
        return model_response.status_code == 200
    except Exception as e:
        logger.error(f"Ollama validation failed: {str(e)}")
        return False

@router.post("")
async def chat(request: ChatRequest):
    """Chat endpoint with persistent memory and context management"""
    try:
        # get or create session
        session = chat_sessions.get(request.session_id) if request.session_id else create_chat_session()
        logger.info(f"Processing chat request for session: {session.session_id}")

        # process query through supervisor agent
        agent_response = await agent_service.process_chat_query(
            query=request.prompt,
            context={
                "session_id": session.session_id,
                "chat_history": [msg.model_dump() for msg in session.messages],
                # **request.context if request.context else {},
            }
        )

        # Create messages
        user_message = ChatMessage(role="user", content=request.prompt)
        assistant_message = ChatMessage(role="assistant", content=agent_response["response"])


        # session.messages.append(user_message)
        # Update session
        session.messages.extend([user_message, assistant_message])
        session.last_updated = get_utc_now()  # Using timezone-aware time
        session.metadata.update({
            "last_intent": agent_response["intent"],
            "last_confidence": agent_response["confidence"],
            "processing_steps": agent_response["processing_steps"]
        })

        chat_sessions[session.session_id] = session
        
        return {
            "session_id": session.session_id,
            "response": agent_response["response"],
            "history": [msg.model_dump() for msg in session.messages],
            "metadata": {
                **session.metadata,
                "intent": agent_response["intent"],
                "confidence": agent_response["confidence"]
            }
        }

    except Exception as e:
        logger.error(f"Chat processing error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Chat processing failed",
                "message": str(e)
            }
        )

@router.get("/{session_id}")
async def get_chat_history(session_id: str):
    """Retrieve chat history for a specific session"""
    logger.info(f"Retrieving chat history for session: {session_id}")
    session = chat_sessions.get(session_id)
    if not session:
        logger.warning(f"Session not found: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session.session_id,
        "history": [msg.model_dump() for msg in session.messages],
        "metadata": session.metadata,
        "created_at": session.created_at,
        "last_updated": session.last_updated
    }