# Step 1: Setup pydantic model
from pydantic import BaseModel
from typing import List
from agent import get_response_from_ai_agent

class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool


# Step 2: Setup ai agent from FrontEnd Request

from fastapi import FastAPI
app = FastAPI(title="Langgraph AI agent")

ALLOWED_MODEL_NAMES = ["llama3-70b-8192", "mixtral-8x7b", "llama-3.3-70b-versatile", "gpt-4o-mini"]

@app.post("/chat")
def chatEndpoint(request: RequestState):
    """
    API Endpoint to interact with the chatbot using LangGraph and search tools
    It dynammically selects the model specified in the request 
    """

    llm_id = request.model_name
    query = request.messages
    allow_search = request.allow_search
    system_prompt = request.system_prompt
    provider = request.model_provider


    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error": "Invalid model name. Kindly select a valid model"}
    
    # Create AI agent and get response from it
    response = get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)



# Step 3: Run app and explore swagger UI Docs

