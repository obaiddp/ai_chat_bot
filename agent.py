import os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage


# API Keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Setup LLM
openai_llm = ChatOpenAI(model="gpt-4o-mini")
groq_llm = ChatGroq(model="llama-3.3-70b-versatile")


def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):

    if (provider == "Groq"):
        llm = llm_id
    elif (provider == "OpenAI"):
        llm = llm_id

    # Tavily web search
    search_tool = [TavilySearchResults(max_results=2)] if allow_search else []

    # Create AI agent
    agent = create_react_agent(
        model=groq_llm,
        tools=search_tool
    )

    # Pass system prompt as a message
    state = {"messages": [HumanMessage(content=f"{system_prompt}\n{query}")]}
    response = agent.invoke(state)


    # Extract AI messages
    messages = response.get("messages", [])  # Ensure messages exist to avoid errors
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]

    # Print the last AI message safely
    if ai_messages:
        return ai_messages[-1]
    else:
        return "No AI response found."













# System Prompt and Query
system_prompt = "Act as a smart and friendly AI chatbot."
query = "Tell me about cooking rabbit."


#print(response)


