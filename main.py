# main.py
from fastapi import FastAPI, Request, HTTPException
import httpx
from pydantic import BaseModel
import requests

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello World"}

class MessageRequest(BaseModel):
    message: str
    model: str = "llama3.2:latest" 

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

