import os
import logging
from typing import Literal, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from services.llm_agents import classify_message, retrieve_knowledge, generate_draft_response
from core.data_models import AgentState, SupportResponse, RoutingDecision

logger = logging.getLogger(__name__)

# Define the Decision node 
def route_request(state: AgentState) -> Literal["retrieve", "handoff"]:
    """
    Conditional edge: Determines the next step based on the classified urgency level.
    """
    analysis: RoutingDecision = state['analysis']
    urgency = analysis.get('urgency', 'Low').lower()
    
    # High/Critical -> immediate human intervention.
    if urgency in ["critical", "high"]:
        logger.warning(f"Urgency is {urgency}. Skipping RAG and Generation. Proceeding to Handoff.")
        return "handoff"
    
    # Low/Medium -> RAG and Automatic Generation.
    logger.info(f"Urgency is {urgency}. Proceeding with RAG.")
    return "retrieve"

# Define Handoff Node (Critical/High) 
def human_handoff(state: AgentState) -> AgentState:
    """
    Finalizes the state for human escalation when urgency is Critical or High.
    """
    analysis: RoutingDecision = state['analysis']
    
    # Create a quick response for the client acknowledging the escalation
    handoff_message = (
        f"ISSUE ESCALATED: Due to the high urgency level ('{analysis['urgency']}') "
        f"and topic ('{analysis['topic']}'), this request is immediately being routed to a human agent. "
        f"Expected response time is shorter than usual. Ticket created."
    )
    
    # Set the final draft response to the escalation message
    state["draft_response"] = handoff_message
    
    return state

# Build the workflow 
def build_workflow():
    """Builds and compiles the support router workflow using LangGraph."""
    
    # Define the graph state 
    workflow = StateGraph(AgentState)
    
    # Define nodes
    workflow.add_node("classify", classify_message)
    workflow.add_node("retrieve", retrieve_knowledge)
    workflow.add_node("generate", generate_draft_response)
    workflow.add_node("handoff", human_handoff)

    # Set the entry point
    workflow.set_entry_point("classify")

    # Add edges
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    workflow.add_edge("handoff", END)
    
    workflow.add_conditional_edges(
        "classify",
        route_request, 
        {
            "retrieve": "retrieve", # Go to RAG
            "handoff": "handoff"    # Go to Handoff
        }
    )

    # Compile the graph
    app = workflow.compile()
    
    logger.info("Support Router Workflow compiled successfully.")
    return app

# Main execution function 
router_app = build_workflow()

async def run_support_workflow(message: str, customer_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes the compiled LangGraph workflow.
    """
    # Define initial state
    initial_state: AgentState = {
        "request": {
            "request_id": str(os.urandom(4).hex()), # Short ID for logging
            "customer_id": customer_id or "anonymous",
            "message": message
        },
        "analysis": None,
        "context": [],
        "draft_response": None
    }
    
    # Run the graph
    final_state: AgentState = await router_app.ainvoke(initial_state)
    
    # Return only the essential parts for the API response
    return {
        "analysis": final_state['analysis'],
        "draft_response": final_state['draft_response']
    }