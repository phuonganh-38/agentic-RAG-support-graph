import os
import logging
from langfuse.langchain import CallbackHandler
from langchain.globals import set_llm_cache, get_llm_cache
from langchain_community.cache import SQLiteCache
from dotenv import load_dotenv

# Load environments
load_dotenv()

logger = logging.getLogger(__name__)

# --- LANGFUSE SETUP ---
def get_langfuse_callback():
    return CallbackHandler()

# --- CACHING SETUP ---
def setup_llm_caching(cache_path: str = "./data/llm_cache.db") -> None:
    try:
        if get_llm_cache() is None:
            set_llm_cache(SQLiteCache(database_path=cache_path))
            logger.info(f"LangChain LLM Cache enabled at: {cache_path}")
    except Exception as e:
        logger.error(f"Failed to set up LLM caching: {e}")
