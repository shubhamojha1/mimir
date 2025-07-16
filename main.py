# main.py
from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware

import httpx
from pydantic import BaseModel
import requests
from typing import Dict, List

# from api.intent import router as intent_router
from backend.api.upload import router as upload_router

app = FastAPI(
    title = "Mimir autonomous research and study assistant",
    description = "it does something",
    version="0.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False, #TODO: change to True
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include the upload router
app.include_router(upload_router, tags=["upload"])

@app.get("/")
async def read_root():
    return {"message": "Hello World"}

class MessageRequest(BaseModel):
    message: str
    model: str = "llama3.2:latest" 
    # model: str = "deepseek-r1:7b"

OLLAMA_API_URL = "http://localhost:11434/api/generate"

@app.post("/ask")
def ask_ollama(request: MessageRequest):
    payload = {
        "model": request.model,
        "prompt": request.message,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail={
                "error": "Ollama server error",
                "status_code": 404,
                "ollama_response": response.text
            })
        response.raise_for_status()
        result = response.json()
        return {"response": result.get("response")}
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

# In-memory store for chat history (for demo purposes)
chat_histories: Dict[str, List[Dict[str, str]]] = {}

@app.post("/chat-with-memory")
async def chat_with_memory(
    session_id: str = Body(...),
    prompt: str = Body(...),
    model: str = Body(default="llama3:2b")
):
    # Get or create chat history for this session
    history = chat_histories.get(session_id, [])
    history.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": history,
        "stream": False
    }
    ollama_url = "http://localhost:11434/api/chat"
    async with httpx.AsyncClient() as client:
        response = await client.post(ollama_url, json=payload)
        if response.status_code == 200:
            result = response.json()
            # Add assistant's reply to history
            assistant_reply = result.get("message", {}).get("content")
            if assistant_reply:
                history.append({"role": "assistant", "content": assistant_reply})
            chat_histories[session_id] = history
            return {"response": assistant_reply, "history": history}
        else:
            return {"error": "Ollama server error", "status_code": response.status_code, "ollama_response": response.text}

