import os
import logging
from typing import List, Optional, Dict, Any
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)

class SupportVectorStore:
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = persist_directory
        self.embedding_function = OpenAIEmbeddings()

        # Create the persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize the vector store
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_function,
            collection_name="support_knowledge_base"
        )
        logger.info(f"Vector Store initialized at {persist_directory}")

    def add_documents(self, pdf_path: str, topic: str) -> None:
        try: 
            # Load pdf
            loader = PyPDFLoader(pdf_path)
            docs = loader.load_and_split()

            # Add tag to documents, allows us to filter by topic later
            for doc in docs:
                doc.metadata["topic"] = topic
                doc.metadata["source"] = os.path.basename(pdf_path)

            self.db.add_documents(docs)
            logger.info(f"Successfully added {len(docs)} pages from {pdf_path} under topic '{topic}'")
        except Exception as e:
            logger.error(f"Failed to add documents from {pdf_path}: {e}")
            raise e
    
    def retrieve_solution(self, query: str, topic: Optional[str] = None, k: int = 3) -> List[Document]:
        """
        Search for relevant documents. Uses Metadata Filtering if a topic is provided.
        
        Args:
            query: The customer's message or problem description.
            topic: (Optional) The classified topic from the Classifier Agent.
                   If provided, we limit search to this topic only.
            k: Number of documents to return.
            
        Returns:
            List of relevant Documents.
        """

        try:
            search_kwargs = {"k": k}

            # Smart feature: metadata filtering
            if topic and topic != "General Inquiry":
                search_kwargs["filter"] = {"topic": topic}
                logger.info(f"Searching for '{query}' strictly within topic: {topic}")
            else:
                logger.info(f"Searching for '{query}' across ALL topics")

            # Perform Similarity Search
            results = self.db.similarity_search(query, **search_kwargs)

            return results
        
        except Exception as e:
            logger.error(f"Error retrieving solution: {e}")
            return [] # Return empty list
        
    def get_retriever(self):
        return self.db.as_retriever()

