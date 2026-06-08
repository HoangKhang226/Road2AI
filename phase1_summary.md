# Phase 1 Status Report — Thu Thập & Chunking Dữ Liệu

**Ngày hoàn thành**: 08/06/2026  
**Thời gian thực hiện**: ~3 giờ

---

## ✅ Đã Hoàn Thành

### 1. Infrastructure Setup

- ✅ Tạo cấu trúc package `src/aiguru/` theo chuẩn Python
- ✅ Setup `.venv` với Windows Python 3.11.7
- ✅ Cài đặt dependencies: `datasets`, `huggingface_hub`, `numpy`, `pandas`, `pyarrow`
- ✅ Editable install: `pip install -e .`
- ✅ Tạo `.gitignore` cho artifacts
- ✅ Tạo `pyproject.toml` chuẩn

### 2. Module Phase 1

Đã tạo 5 modules hoàn chỉnh trong `src/aiguru/phase1/`:

| Module | File | LOC | Status |
|--------|------|-----|--------|
| Config | `config.py` | 68 | ✅ Working |
| Schema | `metadata_schema.py` | 122 | ✅ Working |
| Collector | `collect.py` | 221 | ⚠️ Partial |
| Chunker | `chunk.py` | 217 | ⚠️ Blocked |
| Entry scripts | `scripts/run_phase1_*.py` | 8 | ✅ Working |

### 3. HuggingFace Dataset Discovery

Đã tìm ra 2 dataset public chính thức từ `tmquan`:

- **Pháp điển**: `tmquan/phapdien-moj-gov-vn` (5 likes, 1458 downloads)
- **Án lệ**: `tmquan/anle-toaan-gov-vn` (8 likes, 7328 downloads)

Collector đã load thành công cả 2 datasets.

### 4. Raw Data Collection

```
raw_data/
├── legal_docs_raw.jsonl          (0 records) ❌
├── precedents_raw.jsonl          (1906 records) ⚠️
└── collection_report.json
```

---

## ❌ Vấn Đề Phát Hiện — BLOCKER

### Triệu Chứng

```json
{
  "total_chunks": 0,
  "total_precedents": 1906,
  "missing_doc_id": 1906,
  "sme_score": 0.0
}
```

**Chunker không tạo ra chunk nào dù có 1906 precedent records.**

### Root Cause Analysis

Dataset `tmquan` có cấu trúc **rất phức tạp** cho NLP research:

```json
{
  "doc_id": "TAND192004",
  "title": "Bản án số: 38/2021/DS-PT",
  "markdown": "...full text...",
  "sentences": [
    {
      "sentence_id": "TAND192004#sen_0001",
      "text": "...",
      "paragraph_id": "...",
      "section_kind": "header"
    }
  ],
  "extracted_json": {
    "entities": [...],
    "statute_refs": [
      {"article": 147, "clause": null, "span": [619, 627]}
    ]
  },
  "applied_article_number": 147,
  "principle_text": null
}
```

**Collector hiện tại** dùng `first_non_empty()` để tìm:
- `["text", "content", "noi_dung", "raw_text", ...]`

Nhưng dataset thực tế có:
- `markdown` (full text)
- `sentences` (array of sentence objects)
- Nested structure sâu 3-4 levels

→ Không match bất kỳ field nào → `raw_text = ""`  
→ Chunker nhận empty text → **0 chunks**

### Phapdien Dataset Issue

Count = 0 có thể do:
1. Dataset chỉ có split khác (không phải "train")
2. Hoặc có filter điều kiện mà collector không pass
3. Cần investigate structure

---

## 🔍 Đề Xuất Giải Pháp

### Option 1: Adapt Collector (Recommended)

**Ưu điểm**: Tận dụng dataset chất lượng cao, đã có 7K+ downloads  
**Nhược điểm**: Cần refactor collector logic

**Action items**:
1. Load 1 sample từ mỗi dataset
2. Print full keys/structure
3. Rewrite `normalize_legal_doc()` và `normalize_precedent()`:
   - Precedents: extract từ `markdown` field
   - Legal docs: investigate actual structure
4. Preserve entity annotations nếu có
5. Map `applied_article_number` → metadata

**Estimated time**: 2-3 giờ

### Option 2: Find Simpler Datasets

Search HuggingFace cho:
- `hoangquang27/data_phapdienvn`
- `Thanh271001/PHAPDIEN`
- Các dataset khác có structure đơn giản hơn

**Ưu điểm**: Nhanh, ít code change  
**Nhược điểm**: Dataset nhỏ hơn, chất lượng không rõ

### Option 3: Manual Test Corpus

Tạo 50-100 chunks thủ công từ:
- Luật Doanh nghiệp (download từ vbpl.vn)
- Luật Hỗ trợ DNNVV
- Parse manual → chuẩn hóa → JSONL

**Ưu điểm**: Kiểm soát hoàn toàn, test pipeline end-to-end  
**Nhược điểm**: Tốn thời gian, không scale cho production

---

## 📊 Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Venv setup | ✅ | ✅ | Done |
| Package structure | ✅ | ✅ | Done |
| Dataset loaded | ✅ | ✅ | Done |
| Raw docs collected | 2000+ | 1906 | ⚠️ Partial |
| Chunks generated | 2000+ | **0** | ❌ Blocked |
| Metadata valid | 100% | N/A | Blocked |

---

## ⏭️ Next Steps

**IMMEDIATE (Unblock Phase 1)**:

1. **Investigate dataset structure** (30 min)
   ```bash
   python -c "from datasets import load_dataset; ds = load_dataset('tmquan/anle-toaan-gov-vn', split='train'); print(ds[0].keys()); print(ds[0])"
   ```

2. **Fix collector extraction** (2 hours)
   - Add `markdown` field support
   - Add `sentences` concatenation fallback
   - Test with 10 samples

3. **Validate chunks** (30 min)
   - Run chunker
   - Inspect `chunks.jsonl`
   - Check `formatted_doc`/`formatted_article` compliance

**DEFERRED (Phase 2)**:

- BM25 index building
- FAISS embedding
- Reranker integration

---

## 🚨 Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Dataset structure incompatible | HIGH | 🔴 Critical | Option 1: Adapt collector |
| Phapdien dataset empty | MEDIUM | 🟡 High | Use Thanh271001 alternative |
| Metadata format wrong vs BTC | MEDIUM | 🟡 High | Verify with sample submission early |
| SME scoring inaccurate | LOW | 🟢 Low | Tune keywords in config |

---

## 📝 Lessons Learned

1. **Always inspect dataset structure first** before writing extraction logic
2. HuggingFace datasets for research often have complex nested schemas
3. `datasets` library needs binary wheels → Windows Python > MSYS Python
4. Editable install (`pip install -e .`) essential for rapid iteration

---

## 🎯 Definition of Done for Phase 1

- [ ] Collector extracts non-empty `raw_text` from both datasets
- [ ] Chunker produces >100 chunks with valid metadata
- [ ] Random sample of 10 chunks reviewed manually
- [ ] `formatted_doc` matches BTC example format
- [ ] `chunks.jsonl` ready for Phase 2 indexing

**Current completion**: **60%** (infrastructure ✅, data extraction ❌)
