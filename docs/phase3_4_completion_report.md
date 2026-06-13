# Phase 3-4 Completion Report

## Scope

Phase 3 and 4 connect the persistent Qdrant hybrid retriever from Phase 2 to a
grounded legal answer generator and crash-safe competition output.

## Phase 3: Batched Unsloth Generation

- Default model: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- Unsloth is loaded before LlamaIndex/Transformers so its VRAM patches apply.
- Questions are generated in configurable GPU batches.
- Retrieved chunks are grouped by legal document and sorted by article number.
- The prompt requires grounded answers, explicit `Điều X` citations, practical
  SME guidance, and the standard AI limitation warning.
- Small raw RRF scores are converted to a relative 0-1 scale for Phase 4.

Configuration is available through environment variables in
`src/aiguru/phase3/config.py` or runner arguments.

## Phase 4: Post-Processing and Resume

- Extracts article citations with tolerant Vietnamese regex matching.
- Builds `relevant_articles` and deduplicated `relevant_docs` from retrieved
  chunk metadata only, preventing hallucinated citations from entering IR output.
- Synchronizes every selected `relevant_article` into the generated answer so
  the organizer's `Điều X` extractor sees the same grounded citation set.
- Appends and flushes one independent JSON object per question to
  `output/results_partial.jsonl`.
- Automatically skips completed question IDs after a crash or restart, ignores
  a partially written final line, and deduplicates IDs during export.
- Atomically exports sorted entries to `output/results.json`.

## Run

```bash
pip install -r requirements-gpu.txt
python scripts/run_phase2_retrieve.py
python scripts/run_phase3_4.py data/R2AIStage1DATA.json --batch-size 4
python scripts/validate_submission.py
```

Re-run the same command to resume. Reduce `--batch-size` if GPU VRAM is limited.

## Verification

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src scripts tests
```

The local test suite covers document-aware context formatting, article
extraction, RRF score normalization, legal-reference fallback, JSONL flush,
resume deduplication, and final JSON export.
