"""
Vector store module for Qdrant management.
Handles embedding storage and retrieval for any module.
Supports Qdrant Cloud, Docker-based Qdrant, and in-memory mode.
"""

from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from config import ModelConfig, get_openai_api_key, get_qdrant_config


# Qdrant connection settings
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
EMBEDDING_DIMENSION = 1536  # text-embedding-3-small dimension


def _is_qdrant_cloud_available() -> bool:
    """Check if Qdrant Cloud is reachable."""
    try:
        config = get_qdrant_config()
        client = QdrantClient(
            url=config["url"],
            api_key=config["api_key"],
            timeout=5
        )
        client.get_collections()
        return True
    except Exception:
        return False


def _is_local_qdrant_available(host: str = QDRANT_HOST, port: int = QDRANT_PORT) -> bool:
    """Check if local Qdrant server is reachable."""
    try:
        client = QdrantClient(host=host, port=port, timeout=2)
        client.get_collections()
        return True
    except Exception:
        return False


class VectorStoreManager:
    """
    Manages the Qdrant vector store for document embeddings.
    Supports multiple modules with separate collections.
    Priority: Qdrant Cloud > Local Docker > In-memory
    """
    
    def __init__(
        self,
        collection_name: str,
        embedding_model: str = ModelConfig.EMBEDDING_MODEL,
        qdrant_host: str = QDRANT_HOST,
        qdrant_port: int = QDRANT_PORT
    ):
        """
        Initialize the vector store manager.
        
        Args:
            collection_name: Name of the Qdrant collection
            embedding_model: OpenAI embedding model to use
            qdrant_host: Qdrant server host (for local mode)
            qdrant_port: Qdrant server port (for local mode)
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._vector_store: Optional[QdrantVectorStore] = None
        self._client: Optional[QdrantClient] = None
        
        # Determine connection mode: cloud > local > memory
        self._qdrant_config = get_qdrant_config()
        self._use_cloud = _is_qdrant_cloud_available()
        self._use_local = not self._use_cloud and _is_local_qdrant_available(qdrant_host, qdrant_port)
        self._use_memory = not self._use_cloud and not self._use_local
    
    def _get_client(self) -> QdrantClient:
        """Get or create the Qdrant client."""
        if self._client is None:
            if self._use_cloud:
                # Qdrant Cloud
                self._client = QdrantClient(
                    url=self._qdrant_config["url"],
                    api_key=self._qdrant_config["api_key"]
                )
            elif self._use_local:
                # Local Docker
                self._client = QdrantClient(
                    host=self.qdrant_host,
                    port=self.qdrant_port
                )
            else:
                # In-memory fallback
                self._client = QdrantClient(":memory:")
        return self._client
    
    def _get_embeddings(self) -> OpenAIEmbeddings:
        """Get or create the embeddings instance."""
        if self._embeddings is None:
            api_key = get_openai_api_key()
            self._embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_key=api_key
            )
        return self._embeddings
    
    def is_memory_mode(self) -> bool:
        """Check if running in memory mode."""
        return self._use_memory
    
    def is_cloud_mode(self) -> bool:
        """Check if running with Qdrant Cloud."""
        return self._use_cloud
    
    def get_connection_mode(self) -> str:
        """Get current connection mode."""
        if self._use_cloud:
            return "cloud"
        elif self._use_local:
            return "local"
        return "memory"
    
    def vector_store_exists(self) -> bool:
        """Check if the collection already exists in Qdrant."""
        try:
            client = self._get_client()
            collections = client.get_collections().collections
            return any(c.name == self.collection_name for c in collections)
        except Exception:
            return False
    
    def _ensure_collection_exists(self) -> None:
        """Create the collection if it doesn't exist."""
        client = self._get_client()
        if not self.vector_store_exists():
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE
                )
            )
    
    def create_vector_store(self, documents: List[Document]) -> QdrantVectorStore:
        """Create a new vector store from documents."""
        embeddings = self._get_embeddings()
        client = self._get_client()
        
        # Delete existing collection if it exists (for fresh rebuild)
        if self.vector_store_exists():
            client.delete_collection(self.collection_name)
        
        if self._use_cloud:
            # Qdrant Cloud mode
            self._vector_store = QdrantVectorStore.from_documents(
                documents=documents,
                embedding=embeddings,
                url=self._qdrant_config["url"],
                api_key=self._qdrant_config["api_key"],
                collection_name=self.collection_name
            )
        elif self._use_local:
            # Local Docker mode
            self._vector_store = QdrantVectorStore.from_documents(
                documents=documents,
                embedding=embeddings,
                url=f"http://{self.qdrant_host}:{self.qdrant_port}",
                collection_name=self.collection_name
            )
        else:
            # In-memory mode
            self._vector_store = QdrantVectorStore.from_documents(
                documents=documents,
                embedding=embeddings,
                collection_name=self.collection_name,
                location=":memory:"
            )
        
        return self._vector_store
    
    def load_vector_store(self) -> QdrantVectorStore:
        """Load an existing vector store from Qdrant."""
        embeddings = self._get_embeddings()
        
        if self._use_cloud:
            # Qdrant Cloud mode
            self._vector_store = QdrantVectorStore.from_existing_collection(
                embedding=embeddings,
                url=self._qdrant_config["url"],
                api_key=self._qdrant_config["api_key"],
                collection_name=self.collection_name
            )
        elif self._use_local:
            # Local Docker mode
            self._vector_store = QdrantVectorStore.from_existing_collection(
                embedding=embeddings,
                url=f"http://{self.qdrant_host}:{self.qdrant_port}",
                collection_name=self.collection_name
            )
        else:
            # In-memory mode: can't load existing
            raise ValueError("Cannot load existing store in memory mode")
        
        return self._vector_store
    
    def get_or_create_vector_store(
        self,
        documents: Optional[List[Document]] = None
    ) -> QdrantVectorStore:
        """Get existing vector store or create a new one."""
        if self._vector_store is not None:
            return self._vector_store
        
        # In memory mode, always need to create fresh
        if self._use_memory:
            if documents is None:
                raise ValueError(
                    "Mode mémoire: documents requis pour créer la base vectorielle."
                )
            return self.create_vector_store(documents)
        
        # Cloud or Local mode: can load existing collections
        if self.vector_store_exists():
            return self.load_vector_store()
        
        if documents is None:
            raise ValueError(
                "Aucune base vectorielle existante et aucun document fourni."
            )
        
        return self.create_vector_store(documents)
    
    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """Get a retriever from the vector store."""
        if self._vector_store is None:
            raise ValueError("Vector store not initialized.")
        
        if search_kwargs is None:
            search_kwargs = {"k": 8}
        
        return self._vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def similarity_search(self, query: str, k: int = 8) -> List[Document]:
        """Perform similarity search on the vector store."""
        if self._vector_store is None:
            raise ValueError("Vector store not initialized.")
        
        return self._vector_store.similarity_search(query, k=k)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            client = self._get_client()
            info = client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
                "mode": self.get_connection_mode()
            }
        except Exception as e:
            return {"error": str(e)}


def create_vector_store_manager(module_config: Dict[str, Any]) -> VectorStoreManager:
    """
    Factory function to create a VectorStoreManager for a specific module.
    
    Args:
        module_config: Module configuration dictionary
        
    Returns:
        VectorStoreManager: Configured vector store manager instance
    """
    return VectorStoreManager(
        collection_name=module_config["collection_name"]
    )
