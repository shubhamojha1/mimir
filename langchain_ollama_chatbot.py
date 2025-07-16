from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from fastapi import FastAPI, Request
from pydantic import BaseModel

llm = Ollama(
    base_url="http://localhost:11434",
    model="llama3.2:latest"
)

# Comprehensive system prompt for the agent
SYSTEM_PROMPT = (
    "You are an intelligent, helpful, and friendly AI assistant. "
    "You can answer questions, provide explanations, help with coding, and hold engaging conversations. "
    "Be concise, clear, and polite. If you do not know the answer, say so honestly. "
    "If the user asks for code, provide well-formatted and correct code snippets. "
    "If the user asks for advice, be thoughtful and unbiased. "
    "Always maintain a positive and professional tone. "
    "If the user asks about your capabilities, explain that you are powered by a local Llama 3.2b model via Ollama and LangChain. "
)

# Set up memory for the agent
memory = ConversationBufferMemory(memory_key="chat_history")

def simple_tool(input: str) -> str:
    return f"You said: {input}"

tools = [
    Tool(
        name="EchoTool",
        func=simple_tool,
        description="Echoes the user's input."
    )
]

# Initialize the agent with memory
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

def chat_with_agent(user_input: str) -> str:
    # Prepend the system prompt to the conversation if it's the first message
    if not memory.buffer:
        memory.save_context({"input": ""}, {"output": SYSTEM_PROMPT})
    response = agent.run(user_input)
    return response

app = FastAPI()

class ChatRequest(BaseModel):
    user_input: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    reply = chat_with_agent(request.user_input)
    return {"response": reply}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        import uvicorn
        uvicorn.run("langchain_ollama_chatbot:app", host="127.0.0.1", port=8000, reload=True)
    else:
        print("Chatbot with memory (Ollama + Llama 3.2b)")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            reply = chat_with_agent(user_input)
            print(f"Bot: {reply}")
