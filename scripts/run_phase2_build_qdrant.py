"""
AIGuru Phase 2 - Build Qdrant Index Script

Build persistent Qdrant vector database incrementally from chunks.
Unlike FAISS, this persists to disk and doesn't need rebuild from scratch.
"""

import sys
import json
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from aiguru.phase2.vector_db import VectorDBManager, chunks_to_nodes
from aiguru.paths import KNOWLEDGE_DIR

def main():
    """Build Qdrant index from chunks.jsonl."""
    print("=" * 60)
    print("PHASE 2: BUILDING QDRANT INDEX")
    print("=" * 60)
    
    # Load chunks
    chunks_file = KNOWLEDGE_DIR / "chunks.jsonl"
    print(f"\n[1/3] Loading chunks from {chunks_file}...")
    
    chunks = []
    with chunks_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    
    print(f"✅ Loaded {len(chunks)} chunks")
    
    # Initialize embedding model
    print(f"\n[2/3] Initializing embedding model...")
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3",
        device="cpu",
    )
    print(f"✅ Model loaded: BAAI/bge-m3 on CPU")
    
    # Initialize vector DB
    print(f"\n[3/3] Building Qdrant index...")
    vector_db = VectorDBManager(
        embedding_model=embed_model,
        collection_name="aiguru_legal",
    )
    
    # Convert chunks to nodes
    print(f"Converting chunks to LlamaIndex nodes...")
    nodes = chunks_to_nodes(chunks)
    print(f"✅ Created {len(nodes)} nodes")
    
    # Add to vector DB (this will embed and index)
    print(f"\nAdding nodes to Qdrant (this will take time on CPU)...")
    print(f"Progress will be shown during indexing...")
    node_ids = vector_db.add_documents(nodes, show_progress=True)
    
    # Get stats
    stats = vector_db.get_stats()
    
    print("\n" + "=" * 60)
    print("QDRANT INDEX BUILD COMPLETE")
    print("=" * 60)
    print(f"Status: {stats['status']}")
    print(f"Collection: {stats['collection_name']}")
    print(f"Nodes indexed: {stats['num_nodes']}")
    print(f"Storage: {stats['storage_path']}")
    print("\n✅ Index is persistent - no need to rebuild!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
