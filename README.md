# mimir
POST: /ask
body:{
  "message": "What is the capital of India?"
}

command: uvicorn main:app --reload