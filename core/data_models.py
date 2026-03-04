from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Literal, List, Dict, TypedDict, Any
from langchain_core.documents import Document
from .constants import topics, urgency_levels, customer_tones

# Input model
class SupportRequest(BaseModel):
    customer_id: EmailStr = Field(
        ...,
        description="Unique identifier for the customer, expected to be a valid email address."
    )
    message: str = Field(
        ...,
        min_length=30,
        description="The raw content of the customer's support message."
    )

# Internal model (Core)
class RoutingDecision(BaseModel):
    topic: str = Field(
        ...,
        description="The classified topic of the issue."
    )
    urgency: str = Field(
        ...,
        description="The urgency level"
    )
    tone: Literal["Negative", "Neutral", "Positive"] = Field(
        ...,
        description="The emotional tone of the customer messages."
    )
    reasoning: Optional[str] = Field(
        None,
        description="Brief explanation of why AI chose this classification."
    )

# LangGraph
class RequestInfo(TypedDict):
    """Container for the initial request details within the workflow state."""
    request_id: str
    customer_id: str
    message: str

class AgentState(TypedDict):
    """
    The state shared across all nodes in the LangGraph workflow.
    Each node takes the state as input and updates it as output.
    """
    request: RequestInfo
    analysis: Optional[Dict[str, Any]]  # Contains RoutingDecision data (as dict) after classification
    context: List[Document] # Retrieved documents after RAG
    draft_response: Optional[str]


# Output model 
class SupportResponse(BaseModel):
    """
    Schema for the final API response sent back to the client.
    """
    request_id: str = Field(
        ..., 
        description="Unique tracking ID for this specific transaction."
    )
    analysis: RoutingDecision = Field(
        ..., 
        description="The classification details (Topic, Urgency, Tone)."
    )
    draft_response: str = Field(
        ..., 
        description="The AI-generated draft reply for the support agent to review."
    )
