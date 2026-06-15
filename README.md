# Road2AI - Vietnamese Legal RAG

Competition pipeline for R2AI2026 Legal Information Retrieval and Question
Answering. The implementation is optimized for recall-heavy macro F2 while
keeping generated answers grounded in retrieved Vietnamese legal articles.

## Architecture

1. **Phase 1 - Collection and chunking**
   - Collect Pháp điển and precedent data.
   - Recover original instrument IDs and article numbers from source citations.
   - Store competition-ready `formatted_doc` and `formatted_article` metadata.
2. **Phase 2 - Retrieval**
   - Legal regex/phrase BM25.
   - BGE-M3 dense retrieval through persistent Qdrant.
   - RRF fusion, legal-domain query expansion, and optional cross-encoder reranking.
   - Crash-safe retrieval cache for all 2,000 questions.
3. **Phase 3 - Generation**
   - Batched 4-bit Unsloth Qwen2.5-7B inference.
   - Document-separated context and an explicit allow-list of valid citations.
4. **Phase 4 - Post-processing**
   - Remove unsupported `Điều X` citations.
   - Add grounded citation fallback and the standard AI limitation warning.
   - Stream results to JSONL and resume after interruption.
5. **Submission**
   - Strictly validate all 2,000 rows and create a flat `submission.zip`.

## Runtime

Use Python 3.10-3.12. For a CUDA/Colab environment:

```bash
pip install -r requirements-gpu.txt
```

Run the entire pipeline:

```bash
python scripts/run_pipeline.py --device cuda --generation-batch-size 4
```

Artifacts are checkpointed, so completed phases can be skipped:

```bash
python scripts/run_pipeline.py --device cuda --skip-collect --skip-index
```

Use `--reset-index` after changing chunking or metadata logic. Resume an
interrupted generation run with `--skip-collect --skip-index` so completed
retrieval and answer rows are preserved.

Individual competition steps:

```bash
python scripts/run_phase1_collect.py
python scripts/run_phase1_chunk.py
python scripts/run_phase2_build_bm25.py
python scripts/run_phase2_build_qdrant.py --device cuda
python scripts/run_phase2_retrieve.py
python scripts/run_phase3_4.py data/R2AIStage1DATA.json --batch-size 4
python scripts/validate_submission.py
```

The final upload artifact is `output/submission.zip`.

## Verification

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src scripts tests
```

For a hand-labelled development set:

```bash
python scripts/local_eval.py output/results.json data/local_reference.json
```

After Phase 2 retrieval, run the automatic silver check on questions that
explicitly mention `Điều X`:

```bash
python scripts/evaluate_retrieval_cache.py
```

Retune F2 article-selection settings from the cached retrieval results without
rerunning the LLM:

```bash
python scripts/retune_submission.py --min-articles 3 --max-articles 8 \
  --safe-threshold 0.3 --high-conf-threshold 0.5
```

---

# R2AI-MENTOR-DAY2
