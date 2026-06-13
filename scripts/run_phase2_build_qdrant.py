"""
AIGuru Phase 2 - Build Qdrant Index Script

Build persistent Qdrant vector database incrementally from chunks.
Unlike FAISS, this persists to disk and doesn't need rebuild from scratch.
"""

import sys
import json
import argparse
import hashlib
import shutil
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from aiguru.paths import KNOWLEDGE_DIR, STORAGE_DIR


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def main():
    """Build Qdrant index from chunks.jsonl."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from aiguru.phase2.vector_db import VectorDBManager, chunks_to_nodes
    print("=" * 60)
    print("PHASE 2: BUILDING QDRANT INDEX")
    print("=" * 60)
    
    # Load chunks
    chunks_file = KNOWLEDGE_DIR / "chunks.jsonl"
    metadata_dir = STORAGE_DIR / "aiguru_legal"
    manifest_file = metadata_dir / "corpus_manifest.json"
    if args.reset:
        shutil.rmtree(STORAGE_DIR / "qdrant_data", ignore_errors=True)
        shutil.rmtree(metadata_dir, ignore_errors=True)
    corpus_hash = file_sha256(chunks_file)
    if manifest_file.exists():
        previous = json.loads(manifest_file.read_text(encoding="utf-8"))
        if previous.get("chunks_sha256") != corpus_hash:
            raise RuntimeError(
                "chunks.jsonl changed since the existing Qdrant build. "
                "Re-run with --reset to avoid mixing stale and new chunks."
            )
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps({"chunks_sha256": corpus_hash, "status": "building"}, indent=2),
        encoding="utf-8",
    )
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
        device=args.device,
    )
    print(f"✅ Model loaded: BAAI/bge-m3 on {args.device}")
    
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
    print(f"\nAdding nodes to Qdrant (this will take time on {args.device})...")
    print(f"Progress will be shown during indexing...")
    node_ids = vector_db.add_documents(nodes, show_progress=True, batch_size=args.batch_size)
    
    # Get stats
    stats = vector_db.get_stats()
    manifest_file.write_text(
        json.dumps({"chunks_sha256": corpus_hash, "status": "ready", "nodes": stats["num_nodes"]}, indent=2),
        encoding="utf-8",
    )
    
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
