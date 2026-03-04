from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from contextlib import asynccontextmanager
import logging
from infrastructure.monitoring_langfuse import setup_llm_caching
from api.routers import router as support_router

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles initialization logic before the app starts receiving requests.
    """
    # Startup
    logger.info("Starting Smart Support Router...")
    
    # Enable caching (cache DB will be created in ./data/llm_cache.db)
    setup_llm_caching()
    
    yield # Application runs here
    
    # Shutdown
    logger.info("Shutting down Smart Support Router...")


# Initialize app
app = FastAPI(
    title="Smart Support Router API",
    description="AI-powered Customer Support Routing System using LangGraph and RAG.",
    version="1.0.0",
    lifespan=lifespan
)

# Setup Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics setup
# Create a dedicated metrics app to handle /metrics endpoint automatically
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Connect the specific logic paths from routers.py
app.include_router(support_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)