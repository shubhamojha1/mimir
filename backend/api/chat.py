from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime, timezone
import uuid
import httpx
import logging

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
        session = chat_sessions.get(request.session_id) if request.session_id else create_chat_session()
        logger.info(f"Processing chat request for session: {session.session_id}")
        
        user_message = ChatMessage(role="user", content=request.prompt)
        session.messages.append(user_message)
        session.last_updated = get_utc_now()  # Using timezone-aware time
        
        if request.context:
            session.metadata.update(request.context)
            logger.debug(f"Updated context for session {session.session_id}: {request.context}")

        messages = [{"role": msg.role, "content": msg.content} 
                   for msg in session.messages]
        
        payload = {
            "model": request.model,
            "messages": messages,
            "stream": False
        }

        logger.debug(f"Sending request to Ollama with payload: {payload}")
        ollama_url = "http://localhost:11434/api/chat"
        
        async with httpx.AsyncClient(timeout=30.0) as client:  # Added timeout
            # Validate Ollama connection
            if not await validate_ollama_connection(client, request.model):
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "Ollama service unavailable",
                        "message": "Could not connect to Ollama server or model not found"
                    }
                )

            try:
                response = await client.post(ollama_url, json=payload)
                response.raise_for_status()
                
                result = response.json()
                logger.debug(f"Received response from Ollama: {result}")
                
                assistant_reply = result.get("message", {}).get("content")
                
                if not assistant_reply:
                    logger.error(f"Empty response from Ollama. Full response: {result}")
                    raise ValueError("Empty response from Ollama")

                assistant_message = ChatMessage(
                    role="assistant",
                    content=assistant_reply
                )
                session.messages.append(assistant_message)
                chat_sessions[session.session_id] = session

                return {
                    "session_id": session.session_id,
                    "response": assistant_reply,
                    "history": [msg.dict() for msg in session.messages],
                    "metadata": session.metadata
                }

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP Status Error: {e.response.status_code} - {e.response.text}")
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail={
                        "error": "Ollama server error",
                        "message": e.response.text,
                        "status_code": e.response.status_code
                    }
                )
            except httpx.RequestError as e:
                logger.error(f"Request Error: {str(e)}")
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "Cannot connect to Ollama server",
                        "message": str(e)
                    }
                )
            except Exception as e:
                logger.error(f"Unexpected error during chat: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Internal server error",
                        "message": str(e),
                        "type": type(e).__name__
                    }
                )

    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Chat processing failed",
                "message": str(e),
                "type": type(e).__name__
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
        "history": [msg.dict() for msg in session.messages],
        "metadata": session.metadata,
        "created_at": session.created_at,
        "last_updated": session.last_updated
    }