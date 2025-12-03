# router.py
"""
Router for intelligent query routing and chatbot functionality
"""

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import httpx
from models import ChatState
from config import GENAILAB_API_KEY, GENAILAB_BASE_URL, GENAILAB_MODEL
from utils import detect_stock_query, detect_technical_analysis_query, extract_stock_symbol_from_message, extract_analysis_period


# Initialize LLM
client = httpx.Client(verify=False)
llm = ChatOpenAI(
    base_url=GENAILAB_BASE_URL,
    model=GENAILAB_MODEL,
    api_key="sk--17WECoVy-pTpI_dRbdSvQ",
    http_client=client
)


def router_node(state: ChatState) -> ChatState:
    """
    Route user queries to appropriate handlers based on content analysis
    
    Args:
        state: Current chat state
        
    Returns:
        Updated chat state with routing flags
    """
    messages = state["messages"]
    if not messages:
        return state
        
    # Get the last user message
    last_message = messages[-1]
    if not isinstance(last_message, HumanMessage):
        return state
    
    user_message = last_message.content
    state["last_user_message"] = user_message
    
    # Check for technical analysis query first (higher priority)
    if detect_technical_analysis_query(user_message):
        state["needs_technical_analysis"] = True
        state["needs_stock_data"] = False
        state["stock_symbol"] = extract_stock_symbol_from_message(user_message)
        state["analysis_period"] = extract_analysis_period(user_message)
    # Check for stock price query
    elif detect_stock_query(user_message):
        state["needs_stock_data"] = True
        state["needs_technical_analysis"] = False
        state["stock_symbol"] = extract_stock_symbol_from_message(user_message)
        state["analysis_period"] = ""
    else:
        state["needs_stock_data"] = False
        state["needs_technical_analysis"] = False
        state["stock_symbol"] = ""
        state["analysis_period"] = ""
    
    return state


def chatbot_node(state: ChatState) -> ChatState:
    """
    Process user message and generate response for general queries
    
    Args:
        state: Current chat state
        
    Returns:
        Updated chat state with AI response
    """
    # Get the conversation history for context
    messages = state["messages"]
    
    # Call the LLM with full conversation history
    response = llm.invoke(messages)
    
    # Append AI response to messages
    messages.append(AIMessage(content=response.content))
    
    return {"messages": messages}


def route_to_agent(state: ChatState) -> str:
    """
    Determine which node to route to based on user query type
    
    Args:
        state: Current chat state
        
    Returns:
        Name of the agent node to route to
    """
    if state.get("needs_technical_analysis", False):
        return "technical_analysis_agent"
    elif state.get("needs_stock_data", False):
        return "stock_agent"
    else:
        return "chatbot"
