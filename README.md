# Smart Support System

## Introduction

## About LLM Langgraph
Users use this application by entering their customer ID (it must be an email), and message into Terminal interface about their issue. This ticket is sent to a backend API where an AI agent immediately analyses the message to identify the topic, urgency, and emotional tone. Using a workflow managed by LangGraph, the system then decides whether to answer the request using a knowledge base or escalate it to a human for high-priority issues. For automated cases, the system generates a draft response based on the retrieved information. Finally, the complete ticket results — including the classification details and the draft reply — are returned and displayed as a JSON response in Postman.

## Main Functionality
- Smart ticket classification: uses a dedicated AI agent to analyse incoming messages to determine topic (Billing & Subscription, Technical Issue, Account Management, Feature Request, General Inquiry), urgency level (Critical, High, Medium, Low), and tones (Negative, Neutral, Low)
- Dynamic Workflow Routing: Implements branching logic via LangGraph to automatically decide whether a ticket requires immediate human handoff (for high/critical urgency) or can be handled by the AI.
- Langfuse Integration for deep observability: users can access Langfuse to track token usage, cost, latency, median response time, etc.

## Set up the environment

### Prerequisites
- Langfuse account with `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL`
- `OPENAI_API_KEY`

### Installation

1. **Clone the Repository**
    ```
    git clone
    cd
    ```

2. **Create a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```
4. **Set Up Environment Variables**
    Create a `.env` file in the root directory and add the following environment variables:

    ```env
    OPENAI_API_KEY=your_openai_api_key
    LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
    LANGFUSE_SECRET_KEY=your_langfuse_secret_key
    LANGFUSE_BASE_URL = "https://cloud.langfuse.com"
    ```

### Running the Project

## Project structure
- `main.py`: Entry point for the agent handler.
- `app.py`: Core service initializer.
- `llm_agents.py`: Defines 3 agents for Classification, Knowledge Retrieval, and Draft Response Generation.
- `router_workflow.py`: Design the processing flow and branching rules of the application.
- `routers.py`: Build API gateways to receive HTTP requests from users, transfer data between users and the system.
- `support_solutions.txt`: knowledge base
- `ingest.py`: reads and chunks data from `support_solutions.txt`, embeds it and stores in `chroma_db` folder.
- `vector_store.py`: connects to embedded knowledge base (`chroma_db`) and search, retrieve relevant solutions for user requests.
- `monitoring_langfuse.py`: set up langfuse for observability.
- `requirements.txt`: Lists the dependencies required for the project.
- `.env`: Contains environment variables for the project.