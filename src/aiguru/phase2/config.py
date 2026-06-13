"""
AIGuru Legal RAG - Phase 2 Hybrid Retrieval Configuration.

Phase 2 thực hiện:
1. BM25 sparse retrieval (Vietnamese tokenization)
2. Dense retrieval với FAISS (bge-m3 embeddings)
3. RRF fusion để kết hợp kết quả
4. Cross-encoder reranking để tinh chỉnh

Tất cả config cho retrieval pipeline tập trung ở đây.
"""

from aiguru.paths import KNOWLEDGE_DIR
from pathlib import Path

# === PATHS ===

CHUNKS_FILE = KNOWLEDGE_DIR / "chunks.jsonl"
BM25_INDEX_DIR = KNOWLEDGE_DIR / "bm25"
FAISS_INDEX_DIR = KNOWLEDGE_DIR / "faiss"
RETRIEVAL_CACHE_DIR = KNOWLEDGE_DIR / "retrieval_cache"

BM25_CORPUS_FILE = BM25_INDEX_DIR / "corpus.pkl"
BM25_TOKENIZER_CONFIG_FILE = BM25_INDEX_DIR / "tokenizer_config.json"
BM25_CHUNK_ID_MAP_FILE = BM25_INDEX_DIR / "chunk_id_map.json"

FAISS_INDEX_FILE = FAISS_INDEX_DIR / "index.bin"
FAISS_EMBEDDINGS_FILE = FAISS_INDEX_DIR / "embeddings.npy"
FAISS_CHUNK_ID_MAP_FILE = FAISS_INDEX_DIR / "chunk_id_map.json"
FAISS_MODEL_CONFIG_FILE = FAISS_INDEX_DIR / "model_config.json"

# === BM25 CONFIG ===

# Tokenizer cho tiếng Việt
BM25_TOKENIZER = "legal_regex_phrases"

# BM25 parameters
BM25_K1 = 1.5  # Term frequency saturation
BM25_B = 0.75  # Length normalization

# Retrieval config
BM25_TOP_K = 50  # Số lượng documents retrieve từ BM25

# Batch size cho tokenization (tránh memory issues)
BM25_TOKENIZE_BATCH_SIZE = 1000

# === DENSE RETRIEVAL CONFIG ===

# Embedding model candidates (thử theo thứ tự)
EMBEDDING_MODEL_CANDIDATES = [
    "BAAI/bge-m3",  # 568M params, multilingual, excellent for Vietnamese
    "bkai-foundation-models/vietnamese-bi-encoder",  # Vietnamese-specific
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",  # Fallback
]

# Embedding config
EMBEDDING_BATCH_SIZE = 32  # Batch size cho encoding
EMBEDDING_MAX_LENGTH = 512  # Max sequence length
EMBEDDING_NORMALIZE = True  # Normalize embeddings trước khi lưu

# FAISS config
FAISS_INDEX_TYPE = "FlatIP"  # FlatIP (inner product) cho normalized embeddings
FAISS_TOP_K = 50  # Số lượng documents retrieve từ FAISS

# Checkpointing cho embeddings (tránh mất công khi crash)
EMBEDDING_CHECKPOINT_EVERY = 5000  # Save embeddings mỗi N chunks

# === RRF FUSION CONFIG ===

# Reciprocal Rank Fusion parameter
RRF_K = 60  # Constant k trong công thức RRF(d) = Σ 1/(k + rank_i(d))

# Số lượng candidates sau RRF fusion để đưa vào reranker
RRF_TOP_K = 30

# === RERANKER CONFIG ===

# Cross-encoder reranker model candidates
RERANKER_MODEL_CANDIDATES = [
    "BAAI/bge-reranker-v2-m3",  # Best multilingual reranker
    "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Fallback English reranker
]

# Reranker config
RERANKER_BATCH_SIZE = 16
RERANKER_MAX_LENGTH = 512

# === DYNAMIC THRESHOLDING (F2 optimization) ===

# F2 trọng Recall gấp 4 lần Precision → threshold thấp hơn
SAFE_THRESHOLD = 0.3  # Threshold cho context (recall cao)
HIGH_CONF_THRESHOLD = 0.5  # Threshold cho relevant_articles (precision cao hơn)

# Số lượng articles/docs tối đa cho output
MAX_ARTICLES = 10  # Max relevant_articles cho JSON output
MAX_CONTEXT_CHUNKS = 25  # Max chunks cho LLM context

# Fallback khi top score quá thấp
MIN_TOP_SCORE_FOR_NORMAL_MODE = 0.3  # Nếu top-1 < threshold này → safe response mode

# === RETRIEVAL PIPELINE CONFIG ===

# Có enable từng component không (để A/B testing)
ENABLE_BM25 = True
ENABLE_DENSE = True
ENABLE_RRF = True
ENABLE_RERANKER = True

# Weights cho ensemble (nếu không dùng RRF)
BM25_WEIGHT = 0.3
DENSE_WEIGHT = 0.7

# === LOGGING & MONITORING ===

# Cache retrieval results để debug
CACHE_RETRIEVAL_RESULTS = True
RETRIEVAL_CACHE_MAX_SIZE = 1000  # Max số queries cache trong memory

# Log top-K results cho mỗi stage
LOG_TOP_K_PER_STAGE = 5

# === HARDWARE CONFIG ===

# Device cho PyTorch models
DEVICE = "cpu"  # "cuda", "cpu", hoặc "mps" (Apple Silicon)

# Số CPU threads cho BM25 tokenization
NUM_WORKERS = 4

# Memory budget (GB) - warning nếu vượt
MAX_MEMORY_GB = 12  # Colab T4 có 15GB RAM, để 12GB an toàn
