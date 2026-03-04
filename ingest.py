import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
document_path = "./documents/support_solutions.txt"
chroma_db_dir = "./data/chroma_db"
collection_name = "support_knowledge_base"

def get_topic_metadata(doc_content):
    try:
        # Search for the line starting with "Topic:"
        topic_line = next(line for line in doc_content.split('\n') if line.startswith("Topic:"))
        return topic_line.split(":")[1].strip()
    except:
        return "General Inquiry" # Return default if not found

def ingest_data():
    
    if not os.path.exists(document_path):
        logger.error(f"ERROR: Source file not found at {document_path}. Please check the path.")
        return
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.critical("MISSING API KEY: OPENAI_API_KEY not found. Cannot create vectors.")
        return
    
    loader = TextLoader(document_path, encoding="utf-8")
    documents = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    # Attach Topic to each chunk
    for chunk in chunks:
        topic = get_topic_metadata(chunk.page_content)
        chunk.metadata.update({
            "topic": topic,
            "source": os.path.basename(document_path)
        })
        
    logger.info(f"Split {len(documents)} pages into {len(chunks)} chunks.")
    
    # Create Embeddings
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Save to ChromaDB
    try:
        os.makedirs(chroma_db_dir, exist_ok=True)
        
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=chroma_db_dir,
            collection_name=collection_name
        )
        logger.info("Successful!")
    except Exception as e:
        logger.critical(f"CRITICAL ERROR: Failed to save to ChromaDB. Details: {e}")

if __name__ == "__main__":
    ingest_data()