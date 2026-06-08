"""
AIGuru Phase 2 - Build BM25 Index Script

Chạy script này để build BM25 sparse retrieval index từ chunks.jsonl.
Uses simplified tokenizer to avoid hang issues.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from aiguru.phase2.bm25_indexer_simple import main

if __name__ == "__main__":
    main()
