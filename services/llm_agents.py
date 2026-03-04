import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from typing import Dict, List, Any
from langchain.schema import Document
from core.constants import classifier_system_prompt, generator_system_prompt, topics, customer_tones
from core.data_models import RoutingDecision, AgentState
from infrastructure.vector_store import SupportVectorStore
from infrastructure.monitoring import record_llm_usage, count_tokens
from infrastructure.monitoring_langfuse import get_langfuse_callback

logger = logging.getLogger(__name__)

# Initialize models and Vector stores
classifier_model = "gpt-3.5-turbo-0125"
generator_model="gpt-4o-mini"

classifier_llm = ChatOpenAI(
    model=classifier_model,
    temperature=0
)

generator_llm = ChatOpenAI(
    model=generator_model, 
    temperature=0.2
)

# Initialize vector store
vector_store = SupportVectorStore()

# Initialize langfuse for observability
langfuse_handler = get_langfuse_callback()

# Define Agents
# --- Agent 1: Classification ---
def classify_message(state: dict) -> Dict[str, Any]:

    # Define the structured output chain
    parser = JsonOutputParser(pydantic_object=RoutingDecision)

    # Input variables into the prompt
    system_prompt = classifier_system_prompt.format(
        topics=topics,
        customer_tones=customer_tones
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Analyze this message: {message}"),
            ("human", "Format your output strictly as JSON following this schema: {format_instructions}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    classification_chain = prompt | classifier_llm | parser

    # Invoke the chain
    customer_message = state['request']['message']

    try:
        raw_response = classification_chain.invoke({"message": customer_message},
                                                   config={"callbacks": [langfuse_handler]})
        state['analysis'] = raw_response # Update state
        logger.info(
            f"Classification: Topic={raw_response.get('topic')}, "
            f"Urgency={raw_response.get('urgency')}, "
            f"Tone={raw_response.get('tone')}"
        )
    except Exception as e:
        logger.error(f"Classification failed: {e}. Defaulting to General Inquiry.")
        
        # Fallback in case of failure
        fallback_decision = RoutingDecision(
            topic="General Inquiry", 
            urgency="Low", 
            tone="Neutral",
            reasoning=f"Classification model failed due to: {e.__class__.__name__}."
        )

        state['analysis'] = fallback_decision.model_dump()

    return state

# --- Agent 2: Knowledge Retrieval ---
def retrieve_knowledge(state: AgentState) -> AgentState:
    """Retrieves relevant knowledge snippets from the vector store based on the classified topic."""
    
    # Check if retrieval is necessary (Skipped for High/Critical in the workflow, but safer to check)
    urgency = state['analysis'].get('urgency', 'Low')
    if urgency in ["Critical", "High"]:
        logger.warning(f"Skipping RAG for {urgency} urgency.")
        state["context"] = []
        return state
    
    # Get inputs from state
    query = state['request']['message']
    topic = state['analysis'].get('topic')
    
    try:
        # Perform similarity search with metadata filtering
        relevant_docs: List[Document] = vector_store.retrieve_solution(
            query=query, 
            topic=topic, 
            k=3 # Retrieve top 3 chunks
        )
        
        # Update state with retrieved documents
        state["context"] = relevant_docs
        logger.info(f"Retrieved {len(relevant_docs)} documents using filter '{topic}'.")
        
    except Exception as e:
        logger.error(f"Knowledge retrieval failed: {e}")
        state["context"] = [] # Return empty if failure

    return state

# --- Agent 3: Draft Response Generation ---
def generate_draft_response(state: AgentState) -> AgentState:
    """Generates a draft response using the LLM and the retrieved context."""
    
    context_docs: List[Document] = state.get('context', [])
    customer_message = state['request']['message']
    
    # Format context for the generator prompt
    formatted_context = "\n---\n".join([f"Source ({doc.metadata.get('topic')} - {doc.metadata.get('source')}): {doc.page_content}" for doc in context_docs])
    
    # Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", generator_system_prompt),
            ("human", "CONTEXT: {context}\n---\nCUSTOMER MESSAGE: {message}"),
        ]
    )
    generation_chain = prompt | generator_llm | StrOutputParser()
    
    # Invoke the chain and track tokens
    try:
        # Calculate Input Token count
        input_prompt_string = prompt.invoke({"context": formatted_context, "message": customer_message}).to_string()
        input_tokens = count_tokens(input_prompt_string, generator_model)
        
        draft_response = generation_chain.invoke({"context": formatted_context, 
                                                  "message": customer_message},
                                                  config={"callbacks": [langfuse_handler]})
        
        # Calculate Output Token count
        output_tokens = count_tokens(draft_response, generator_model)
        
        # Record usage for monitoring
        record_llm_usage(generator_model, 'input', input_tokens)
        record_llm_usage(generator_model, 'output', output_tokens)

        # Update state
        state["draft_response"] = draft_response
        logger.info(f"Draft response generated. Tokens={input_tokens+output_tokens}")
        
    except Exception as e:
        logger.error(f"Draft generation failed: {e}")
        state["draft_response"] = "System Error: Could not generate a response. Escalating to human support."
        
    return state


