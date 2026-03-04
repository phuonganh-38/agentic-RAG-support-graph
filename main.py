import uvicorn
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point
    """
    try:
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY is not set in environment variables.")

        # Server configuration
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", 8000))
        
        logger.info(f"Starting server on {host}:{port}...")

        # Start server
        is_dev_mode = os.getenv("ENVIRONMENT", "development").lower() == "development"

        uvicorn.run(
            "api.app:app", 
            host=host, 
            port=port, 
            reload=is_dev_mode,
            workers=1
        )

    except Exception as e:
        logger.critical(f"Failed to start the application: {e}")
        raise

if __name__ == "__main__":
    main()