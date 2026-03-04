import os
import time
from typing import Dict, Any
import logging
from prometheus_client import Counter, Histogram, Gauge
from langchain.globals import set_llm_cache, get_llm_cache
from langchain_community.cache import SQLiteCache
import tiktoken
from fastapi import Request

logger = logging.getLogger(__name__)

# Metrics Report
llm_token_counter = Counter(
    'llm_total_tokens_used',
    'Total number of tokens consumed by the LLM (input and output)',
    ['model_name', 'direction']
)

latency = Histogram(
    'router_classification_latency_seconds',
    'Latency of the LLM Classification step',
    ['topic', 'urgency']
)

active_requests = Gauge(
    'router_active_requests',
    'Number of requests currently being processed by the router'
)

# Caching setup


def setup_llm_caching(cache_path: str = "./data/llm_cache.db") -> None:
    try:
        if get_llm_cache() is None:
            set_llm_cache(SQLiteCache(database_path=cache_path))
            logger.info(
                f"LangChain LLM Cache enabled using SQLite at: {cache_path}")
        else:
            logger.info("LangChain LLM Cache already initialized.")
    except Exception as e:
        logger.error(f"Failed to set up LLM caching: {e}")

# Token counting
def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(
            f"Could not count tokens for model {model_name}. Error: {e}")
        return 0


def track_latency(metric: Histogram, labels: Dict[str, Any]):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            active_requests.inc()  # increase

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                # Record duration to the Histogram
                metric.labels(**labels).observe(duration)
                active_requests.dec()  # Decrease
                logger.debug(f"Function {func.__name__} took {duration:.4f}s")

        return wrapper
    return decorator


def record_llm_usage(model: str, input_text: str, output_text: str):
    """Records input and output token usage to the Prometheus Counter."""
    input_tokens = count_tokens(input_text, model)
    output_tokens = count_tokens(output_text, model)

    # Increment global metrics
    llm_token_counter.labels(
        model_name=model, direction='input').inc(input_tokens)
    llm_token_counter.labels(
        model_name=model, direction='output').inc(output_tokens)

    logger.info(
        f"Recorded usage: {input_tokens} input, {output_tokens} output tokens for {model}.")
