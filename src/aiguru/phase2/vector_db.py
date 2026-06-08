"""
AIGuru Legal RAG - Qdrant Vector Database Manager

Simplified implementation inspired by Chat With Data reference code.
Provides persistent vector storage with Qdrant + hybrid retrieval.
"""

from pathlib import Path
import json
from typing import List, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    Document,
    load_index_from_storage,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever


class VectorDBManager:
    """Qdrant-based vector database with hybrid retrieval."""
    
    def __init__(self, embedding_model, collection_name: str = "aiguru_legal"):
        """
        Initialize vector database manager.
        
        Args:
            embedding_model: LlamaIndex embedding model
            collection_name: Qdrant collection name
        """
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        
        # Storage paths
        self.base_storage_dir = Path("storage")
        self.qdrant_path = self.base_storage_dir / "qdrant_data"
        self.index_metadata_dir = self.base_storage_dir / collection_name
        
        # Create directories
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.index_metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Qdrant client (embedded/local mode)
        self._db_client = QdrantClient(path=str(self.qdrant_path))
        
        # Index cache
        self._index = None
    
    def _get_embedding_dimension(self) -> int:
        """Detect embedding dimension from model."""
        try:
            if hasattr(self.embedding_model, "embed_dim"):
                return self.embedding_model.embed_dim
            # Test embedding to get dimension
            sample = self.embedding_model.get_text_embedding("test")
            return len(sample)
        except Exception:
            return 1024  # Default for bge-m3
    
    def _collection_exists(self) -> bool:
        """Check if collection exists in Qdrant."""
        collections = [c.name for c in self._db_client.get_collections().collections]
        return self.collection_name in collections
    
    def _get_storage_context(self) -> StorageContext:
        """Create storage context with Qdrant vector store."""
        # Create collection if not exists
        if not self._collection_exists():
            embed_dim = self._get_embedding_dimension()
            self._db_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
            )
            print(f"✅ Created Qdrant collection '{self.collection_name}' (dim={embed_dim})")
        
        # Create vector store
        vector_store = QdrantVectorStore(
            client=self._db_client,
            collection_name=self.collection_name,
        )
        
        # Try to load existing storage context
        docstore_path = self.index_metadata_dir / "docstore.json"
        if docstore_path.exists():
            try:
                return StorageContext.from_defaults(
                    vector_store=vector_store,
                    persist_dir=str(self.index_metadata_dir),
                )
            except Exception as e:
                print(f"⚠️ Could not load storage context: {e}")
        
        # Create new storage context
        from llama_index.core.storage.docstore import SimpleDocumentStore
        from llama_index.core.storage.index_store import SimpleIndexStore
        
        return StorageContext.from_defaults(
            vector_store=vector_store,
            docstore=SimpleDocumentStore(),
            index_store=SimpleIndexStore(),
        )
    
    def get_index(self) -> Optional[VectorStoreIndex]:
        """Load existing index or return None."""
        if self._index is not None:
            return self._index
        
        docstore_path = self.index_metadata_dir / "docstore.json"
        if not docstore_path.exists():
            return None
        
        try:
            storage_context = self._get_storage_context()
            self._index = load_index_from_storage(
                storage_context,
                embed_model=self.embedding_model,
            )
            print(f"✅ Loaded existing index for '{self.collection_name}'")
            return self._index
        except Exception as e:
            print(f"⚠️ Could not load index: {e}")
            return None
    
    def add_documents(self, nodes: List[TextNode], show_progress: bool = True) -> List[str]:
        """
        Add documents to index.
        
        Args:
            nodes: List of TextNode objects to add
            show_progress: Show progress bar during indexing
            
        Returns:
            List of node IDs that were added
        """
        # Deduplicate input nodes
        unique_nodes = {n.node_id: n for n in nodes}
        nodes_to_add = list(unique_nodes.values())
        
        # Get or create index
        self._index = self.get_index()
        
        if self._index is None:
            print("🚀 Creating new index...")
            storage_context = self._get_storage_context()
            self._index = VectorStoreIndex(
                nodes=nodes_to_add,
                storage_context=storage_context,
                embed_model=self.embedding_model,
                show_progress=show_progress,
                store_nodes_override=True,
            )
        else:
            print("➕ Adding to existing index...")
            # Check for existing nodes
            existing_ids = set(self._index.storage_context.docstore.docs.keys())
            new_nodes = [n for n in nodes_to_add if n.node_id not in existing_ids]
            
            if new_nodes:
                print(f"Inserting {len(new_nodes)} new nodes...")
                self._index.insert_nodes(new_nodes, show_progress=show_progress)
            else:
                print("ℹ️ All nodes already exist in index")
        
        # Persist metadata
        self._index.storage_context.persist(persist_dir=str(self.index_metadata_dir))
        print(f"✅ Index persisted to {self.index_metadata_dir}")
        
        return [n.node_id for n in nodes_to_add]
    
    def get_hybrid_retriever(
        self,
        similarity_top_k: int = 10,
        num_queries: int = 1,
    ):
        """
        Get hybrid retriever (Vector + BM25 with RRF fusion).
        
        Args:
            similarity_top_k: Number of results to return
            num_queries: Number of sub-queries for fusion (1 = use original only)
            
        Returns:
            QueryFusionRetriever or None if index empty
        """
        index = self.get_index()
        if index is None:
            print("❌ No index found. Add documents first.")
            return None
        
        # Get all nodes for BM25
        all_nodes = list(index.storage_context.docstore.docs.values())
        if not all_nodes:
            print("❌ Index is empty")
            return None
        
        # Vector retriever
        vector_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        
        # BM25 retriever
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=all_nodes,
            similarity_top_k=similarity_top_k,
        )
        
        # Hybrid retriever with RRF fusion
        print("🔀 Creating hybrid retriever (Vector + BM25 / RRF)...")
        hybrid_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            similarity_top_k=similarity_top_k,
            num_queries=num_queries,
            mode="reciprocal_rerank",
            use_async=False,
            verbose=True,
        )
        
        return hybrid_retriever
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        index = self.get_index()
        if index is None:
            return {"status": "empty", "num_nodes": 0}
        
        num_nodes = len(index.storage_context.docstore.docs)
        return {
            "status": "ready",
            "collection_name": self.collection_name,
            "num_nodes": num_nodes,
            "storage_path": str(self.index_metadata_dir),
        }


def chunks_to_nodes(chunks: List[dict]) -> List[TextNode]:
    """
    Convert AIGuru chunks to LlamaIndex TextNodes.
    
    Args:
        chunks: List of chunk dicts from Phase 1
        
    Returns:
        List of TextNode objects
    """
    nodes = []
    for chunk in chunks:
        node = TextNode(
            text=chunk["text"],
            id_=chunk["chunk_id"],
            metadata=chunk.get("metadata", {}),
        )
        nodes.append(node)
    return nodes
