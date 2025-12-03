# graph_builder.py
"""
LangGraph workflow builder for the ChatBot application
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from models import ChatState
from router import router_node, chatbot_node, route_to_agent
from stock_agent import stock_agent_node
from technical_analysis_agent import technical_analysis_agent_node


def build_chat_graph():
    """
    Build and compile the LangGraph workflow
    
    Returns:
        Compiled graph with memory persistence
    """
    # Initialize memory checkpointer
    memory = MemorySaver()
    
    # Build the graph with intelligent routing
    graph_builder = StateGraph(ChatState)

    # Add nodes
    graph_builder.add_node("router", router_node)
    graph_builder.add_node("chatbot", chatbot_node) 
    graph_builder.add_node("stock_agent", stock_agent_node)
    graph_builder.add_node("technical_analysis_agent", technical_analysis_agent_node)

    # Define edges with conditional routing
    graph_builder.add_edge(START, "router")
    graph_builder.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "chatbot": "chatbot",
            "stock_agent": "stock_agent",
            "technical_analysis_agent": "technical_analysis_agent"
        }
    )
    graph_builder.add_edge("chatbot", END)
    graph_builder.add_edge("stock_agent", END)
    graph_builder.add_edge("technical_analysis_agent", END)

    # Compile with memory persistence
    graph = graph_builder.compile(checkpointer=memory)
    
    return graph
