# Full Pipeline Upgrade

This upgrade completes the missing competition path from raw legal data to a
validated flat submission ZIP. It improves the repository's likelihood of a
strong leaderboard result, but no implementation can guarantee first place
without leaderboard feedback and a labelled development set.

## Highest-impact changes

- Recover original legal instrument IDs and `Điều X` values from Pháp điển
  `source_note_text`; internal Pháp điển anchors are no longer submitted as law IDs.
- Enforce the organizer's exact `Loại văn bản + Mã văn bản + Trích yếu` prefix.
- Replace whitespace-only BM25 with deterministic legal tokens and phrase bigrams.
- Add domain-aware query expansion, dense retrieval, RRF fusion, candidate
  diversity, SME relevance boost, and official BGE reranker inference.
- Persist retrieval results before loading the LLM, preventing embedding,
  reranker, and generation models from competing for GPU memory; both
  retrieval and generation caches recover from partially written final lines.
- Mark retrieval citations explicitly in the prompt, remove unsupported
  citations from generated answers, and synchronize selected article citations
  into `answer` for the organizer's automatic `Điều X` extraction.
- Add strict 2,000-row validation, flat ZIP packaging, local macro F2
  evaluation, and silver retrieval recall checks.

## Recommended competition loop

1. Run the full pipeline once.
2. Run `python scripts/evaluate_retrieval_cache.py`.
3. Create a manually labelled 30-50 question development set and run
   `python scripts/local_eval.py`.
4. A/B test retrieval and article-selection settings using separate output
   paths or `--reset`.
5. Validate and inspect a sample of answers before using a submission slot.

Useful experiment controls:

```bash
python scripts/run_phase2_retrieve.py --reset --bm25-top-k 75 --fusion-top-k 40
python scripts/run_phase3_4.py data/R2AIStage1DATA.json --reset \
  --min-articles 3 --max-articles 10 --safe-threshold 0.3
python scripts/validate_submission.py
```

Once generation is complete, cheap threshold experiments can reuse both the
retrieval cache and generated answers:

```bash
python scripts/retune_submission.py --min-articles 3 --max-articles 8
```

The current defaults favor recall while retaining a reranker and grounded
generation to limit the corresponding precision loss.
