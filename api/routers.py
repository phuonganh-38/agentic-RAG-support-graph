from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
import uuid
import logging
from core.data_models import SupportRequest, SupportResponse, RoutingDecision
from services.router_workflow import run_support_workflow

# Setup logging
logger = logging.getLogger(__name__)

# Initialize API Router
router = APIRouter()

@router.post("/support/route", response_model=SupportResponse, status_code=200)
async def route_support_request(request: SupportRequest):
    """
    Main Endpoint: Receives a customer message and processes it through the Smart Router.
    
    Steps:
    1. Validates input (handled automatically by Pydantic SupportRequest).
    2. Invokes the LangGraph workflow to Classify -> RAG -> Generate Draft.
    3. Returns the structured response.
    """
    try:
        # Generate a request ID
        request_id = str(uuid.uuid4())
        logger.info(f"Processing request {request_id} for user {request.customer_id}")

        # Execute the core workflow
        workflow_result = await run_support_workflow(
            message=request.message,
            customer_id=request.customer_id
        )

        if not all(k in workflow_result for k in ["analysis", "draft_response"]):
            raise ValueError(
                "LangGraph workflow failed to produce expected keys ('analysis' and 'draft_response'). "
                f"Actual result keys: {list(workflow_result.keys())}"
            )

        # Construct the response
        response = SupportResponse(
            request_id=request_id,
            analysis=RoutingDecision(**workflow_result["analysis"]), # Ensure analysis matches schema
            draft_response=workflow_result["draft_response"]
        )
        
        return response

    except ValidationError as ve:
        logger.error(f"Validation Error for request {request_id}: {ve}")
        raise HTTPException(status_code=422, detail="Internal data validation failed.")
        
    except Exception as e:
        logger.error(f"System Error processing request {request_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error processing support request.")

@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "smart-support-router"}