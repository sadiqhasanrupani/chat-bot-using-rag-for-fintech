"""
Vector Store Service for Finsolve Technologies RAG Chatbot.

This module provides a unified interface for different vector store backends:
1. FAISS (local, file-based)
2. Chroma (local or remote)
3. Qdrant (local or cloud)
4. Pinecone (cloud-based)

Each backend has its own strengths:
- FAISS: Fast local indexing, no external dependencies
- Chroma: Persistent storage, metadata filtering
- Qdrant: Production-ready, high scalability, rich filtering
- Pinecone: Fully managed, high availability, global distribution
"""

import os
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Literal
from pathlib import Path
import logging
from pydantic import BaseModel, Field

# Vector store backends
from langchain_community.vectorstores import FAISS
try:
    from langchain_community.vectorstores import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from langchain_community.vectorstores import Qdrant
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    from langchain_community.vectorstores import Pinecone
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

# Embeddings and documents
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Set up logging
logger = logging.getLogger(__name__)


class VectorStoreType(str, Enum):
    """Types of vector stores supported."""
    FAISS = "faiss"
    CHROMA = "chroma"
    QDRANT = "qdrant"
    PINECONE = "pinecone"


class VectorStoreConfig(BaseModel):
    """Configuration for a vector store."""
    store_type: VectorStoreType = Field(
        default=VectorStoreType.FAISS,
        description="The type of vector store to use"
    )
    
    # Common settings
    collection_name: str = Field(
        default="finsolve",
        description="Name of the collection/index"
    )
    
    # Local storage settings
    persist_directory: Optional[str] = Field(
        default=None,
        description="Directory to persist the vector store (for FAISS and Chroma)"
    )
    
    # Qdrant settings
    qdrant_url: Optional[str] = Field(
        default=None,
        description="URL for Qdrant server (if using Qdrant)"
    )
    qdrant_api_key: Optional[str] = Field(
        default=None,
        description="API key for Qdrant cloud (if using Qdrant cloud)"
    )
    
    # Pinecone settings
    pinecone_api_key: Optional[str] = Field(
        default=None,
        description="API key for Pinecone (if using Pinecone)"
    )
    pinecone_environment: Optional[str] = Field(
        default=None,
        description="Environment for Pinecone (if using Pinecone)"
    )
    
    class Config:
        """Pydantic config."""
        extra = "allow"


class VectorStoreService:
    """Service for managing vector stores with multiple backend options."""
    
    def __init__(
        self,
        config: VectorStoreConfig,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize the vector store service.
        
        Args:
            config: Configuration for the vector store
            embedding_model: Model to use for embeddings
        """
        self.config = config
        
        # Create embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize the vector store
        self.vector_store = None
        self._validate_config()
    
    def _validate_config(self):
        """Validate the configuration based on the selected store type."""
        store_type = self.config.store_type
        
        if store_type == VectorStoreType.CHROMA and not CHROMA_AVAILABLE:
            raise ImportError(
                "Chroma is not available. Install it with: pip install chromadb langchain-chroma"
            )
        
        if store_type == VectorStoreType.QDRANT and not QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant is not available. Install it with: pip install qdrant-client langchain-qdrant"
            )
        
        if store_type == VectorStoreType.PINECONE and not PINECONE_AVAILABLE:
            raise ImportError(
                "Pinecone is not available. Install it with: pip install pinecone-client langchain-pinecone"
            )
        
        # Validate store-specific configurations
        if store_type == VectorStoreType.PINECONE:
            if not self.config.pinecone_api_key or not self.config.pinecone_environment:
                raise ValueError(
                    "Pinecone requires both pinecone_api_key and pinecone_environment"
                )
        
        if store_type == VectorStoreType.QDRANT and self.config.qdrant_url:
            if self.config.qdrant_url.startswith("https://") and not self.config.qdrant_api_key:
                raise ValueError(
                    "Qdrant cloud requires an API key"
                )
    
    def _initialize_faiss(self):
        """Initialize a FAISS vector store."""
        persist_directory = self.config.persist_directory
        
        if persist_directory and os.path.exists(persist_directory):
            logger.info(f"Loading existing FAISS index from {persist_directory}")
            return FAISS.load_local(persist_directory, self.embeddings)
        
        logger.info("Creating new FAISS index")
        return None  # Will be created when documents are added
    
    def _initialize_chroma(self):
        """Initialize a Chroma vector store."""
        if not CHROMA_AVAILABLE:
            raise ImportError("Chroma is not installed")
        
        persist_directory = self.config.persist_directory
        collection_name = self.config.collection_name
        
        logger.info(f"Initializing Chroma with collection {collection_name}")
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
    
    def _initialize_qdrant(self):
        """Initialize a Qdrant vector store."""
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant is not installed")
        
        collection_name = self.config.collection_name
        url = self.config.qdrant_url
        api_key = self.config.qdrant_api_key
        
        if url:
            # Using remote Qdrant
            logger.info(f"Connecting to Qdrant at {url} with collection {collection_name}")
            return Qdrant(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                url=url,
                api_key=api_key,
                prefer_grpc=True
            )
        else:
            # Using local Qdrant
            persist_directory = self.config.persist_directory or ":memory:"
            logger.info(f"Initializing local Qdrant with collection {collection_name}")
            return Qdrant(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                path=persist_directory
            )
    
    def _initialize_pinecone(self):
        """Initialize a Pinecone vector store."""
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone is not installed")
        
        api_key = self.config.pinecone_api_key
        environment = self.config.pinecone_environment
        index_name = self.config.collection_name
        
        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)
        
        # Check if index exists, if not create it
        if index_name not in pinecone.list_indexes():
            logger.info(f"Creating new Pinecone index: {index_name}")
            pinecone.create_index(
                name=index_name,
                dimension=384,  # Dimension for all-MiniLM-L6-v2
                metric="cosine"
            )
        
        logger.info(f"Connecting to Pinecone index: {index_name}")
        return Pinecone.from_existing_index(
            index_name=index_name,
            embedding=self.embeddings
        )
    
    def initialize_vector_store(self):
        """Initialize the vector store based on the configuration."""
        store_type = self.config.store_type
        
        if store_type == VectorStoreType.FAISS:
            self.vector_store = self._initialize_faiss()
        elif store_type == VectorStoreType.CHROMA:
            self.vector_store = self._initialize_chroma()
        elif store_type == VectorStoreType.QDRANT:
            self.vector_store = self._initialize_qdrant()
        elif store_type == VectorStoreType.PINECONE:
            self.vector_store = self._initialize_pinecone()
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")
        
        return self.vector_store
    
    def add_documents(self, documents: List[Document], **kwargs):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            **kwargs: Additional kwargs for the specific vector store
        """
        store_type = self.config.store_type
        
        if self.vector_store is None:
            # For FAISS, we need to create the index with the documents
            if store_type == VectorStoreType.FAISS:
                logger.info(f"Creating new FAISS index with {len(documents)} documents")
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                
                # Save if a persist directory is specified
                if self.config.persist_directory:
                    self.save()
            else:
                # For other stores, initialize first then add documents
                self.initialize_vector_store()
                self.vector_store.add_documents(documents, **kwargs)
        else:
            # Vector store already exists, just add the documents
            logger.info(f"Adding {len(documents)} documents to existing {store_type} store")
            self.vector_store.add_documents(documents, **kwargs)
        
        return self.vector_store
    
    def save(self):
        """Save the vector store if it supports persistence."""
        store_type = self.config.store_type
        persist_directory = self.config.persist_directory
        
        if self.vector_store is None:
            raise ValueError("No vector store to save")
        
        if store_type == VectorStoreType.FAISS and persist_directory:
            logger.info(f"Saving FAISS index to {persist_directory}")
            os.makedirs(persist_directory, exist_ok=True)
            self.vector_store.save_local(persist_directory)
        
        elif store_type == VectorStoreType.CHROMA:
            logger.info("Persisting Chroma collection")
            self.vector_store.persist()
        
        # Qdrant and Pinecone are automatically persisted
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for similar documents in the vector store.
        
        Args:
            query: The query string
            k: Number of results to return
            filter: Filter to apply to the search (if supported by the vector store)
            
        Returns:
            List of documents similar to the query
        """
        if self.vector_store is None:
            self.initialize_vector_store()
            if self.vector_store is None:
                raise ValueError("Vector store is not initialized and no documents are added")
        
        store_type = self.config.store_type
        
        # Different vector stores have different filtering mechanisms
        if store_type == VectorStoreType.FAISS:
            # FAISS doesn't support filtering directly
            results = self.vector_store.similarity_search(query, k=k)
            
            # Manual filtering if filter is provided
            if filter:
                filtered_results = []
                for doc in results:
                    if all(doc.metadata.get(key) == value for key, value in filter.items()):
                        filtered_results.append(doc)
                return filtered_results[:k]
            
            return results
            
        elif store_type == VectorStoreType.CHROMA:
            # Chroma supports where filtering
            if filter:
                where = filter
                return self.vector_store.similarity_search(query, k=k, where=where)
            return self.vector_store.similarity_search(query, k=k)
            
        elif store_type == VectorStoreType.QDRANT:
            # Qdrant uses filter_dict
            if filter:
                filter_dict = filter
                return self.vector_store.similarity_search(query, k=k, filter=filter_dict)
            return self.vector_store.similarity_search(query, k=k)
            
        elif store_type == VectorStoreType.PINECONE:
            # Pinecone uses filter dict
            if filter:
                filter_dict = filter
                return self.vector_store.similarity_search(query, k=k, filter=filter_dict)
            return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 5, 
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """
        Search for similar documents in the vector store with similarity scores.
        
        Args:
            query: The query string
            k: Number of results to return
            filter: Filter to apply to the search (if supported by the vector store)
            
        Returns:
            List of (document, score) tuples
        """
        if self.vector_store is None:
            self.initialize_vector_store()
            if self.vector_store is None:
                raise ValueError("Vector store is not initialized and no documents are added")
        
        store_type = self.config.store_type
        
        if hasattr(self.vector_store, "similarity_search_with_score"):
            # For stores that support similarity_search_with_score directly
            if store_type == VectorStoreType.FAISS:
                results = self.vector_store.similarity_search_with_score(query, k=k)
                
                # Manual filtering if filter is provided
                if filter:
                    filtered_results = []
                    for doc, score in results:
                        if all(doc.metadata.get(key) == value for key, value in filter.items()):
                            filtered_results.append((doc, score))
                    return filtered_results[:k]
                
                return results
            
            elif store_type == VectorStoreType.CHROMA:
                if filter:
                    where = filter
                    return self.vector_store.similarity_search_with_score(query, k=k, where=where)
                return self.vector_store.similarity_search_with_score(query, k=k)
            
            elif store_type == VectorStoreType.QDRANT:
                if filter:
                    filter_dict = filter
                    return self.vector_store.similarity_search_with_score(query, k=k, filter=filter_dict)
                return self.vector_store.similarity_search_with_score(query, k=k)
            
            elif store_type == VectorStoreType.PINECONE:
                if filter:
                    filter_dict = filter
                    return self.vector_store.similarity_search_with_score(query, k=k, filter=filter_dict)
                return self.vector_store.similarity_search_with_score(query, k=k)
        else:
            # Fallback for stores that don't support similarity_search_with_score
            results = self.similarity_search(query, k=k, filter=filter)
            return [(doc, 1.0) for doc in results]  # No score available