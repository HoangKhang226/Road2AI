"""
AIGuru Legal RAG - Phase 1 configuration.
"""

from aiguru.paths import KNOWLEDGE_DIR, RAW_DATA_DIR

PHAPDIEN_DATASET_CANDIDATES = [
    "tmquan/phapdien-moj-gov-vn",
    "phapdien-moj-gov-vn",
    "vietlegal/phapdien-moj-gov-vn",
    "Vietnamese-Legal/phapdien-moj-gov-vn",
]

ANLE_DATASET_CANDIDATES = [
    "tmquan/anle-toaan-gov-vn",
    "anle-toaan-gov-vn",
    "vietlegal/anle-toaan-gov-vn",
    "Vietnamese-Legal/anle-toaan-gov-vn",
]

RAW_LEGAL_DOCS_FILE = RAW_DATA_DIR / "legal_docs_raw.jsonl"
RAW_PRECEDENTS_FILE = RAW_DATA_DIR / "precedents_raw.jsonl"
COLLECTION_REPORT_FILE = RAW_DATA_DIR / "collection_report.json"

CHUNKS_FILE = KNOWLEDGE_DIR / "chunks.jsonl"
CHUNK_STATS_FILE = KNOWLEDGE_DIR / "chunk_stats.json"
METADATA_ERRORS_FILE = KNOWLEDGE_DIR / "metadata_errors.jsonl"

SME_KEYWORDS_HIGH = [
    "doanh nghiệp nhỏ và vừa",
    "dnnvv",
    "sme",
    "hỗ trợ doanh nghiệp",
    "luật doanh nghiệp",
    "thuế giá trị gia tăng",
    "thuế thu nhập doanh nghiệp",
    "hóa đơn điện tử",
    "bảo hiểm xã hội",
    "lao động",
    "kế toán",
    "sở hữu trí tuệ",
    "nhãn hiệu",
    "hợp đồng",
    "thương mại",
]

SME_KEYWORDS_MEDIUM = [
    "doanh nghiệp",
    "công ty",
    "hộ kinh doanh",
    "thuế",
    "hóa đơn",
    "người lao động",
    "báo cáo tài chính",
    "vốn điều lệ",
    "đấu thầu",
    "tín dụng",
    "mặt bằng sản xuất",
]

DOC_TYPE_PATTERNS = {
    "Thông tư liên tịch": ["thông tư liên tịch", "ttlt-"],
    "Nghị quyết liên tịch": ["nghị quyết liên tịch", "nqlt-"],
    "Nghị định": ["nghị định", "nđ-cp", "nd-cp"],
    "Thông tư": ["thông tư", "tt-btc", "tt-blđtbxh", "tt-bkhđt"],
    "Quyết định": ["quyết định", "qđ"],
    "Nghị quyết": ["nghị quyết", "nq-cp", "nq-tw"],
    "Pháp lệnh": ["pháp lệnh"],
    "Bộ luật": ["bộ luật"],
    "Luật": ["luật"],
    "Án lệ": ["án lệ"],
}

MAX_ARTICLE_CHARS_BEFORE_PARAGRAPH_SPLIT = 8000
MIN_CHUNK_CHARS = 80
