from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from aiguru.evaluation import macro_retrieval_metrics, silver_retrieval_recall
from aiguru.phase1.chunk import canonicalize_document_titles, chunk_legal_doc
from aiguru.phase1.collect import normalize_legal_doc
from aiguru.phase1.metadata_schema import normalize_doc_id
from aiguru.phase1.metadata_schema import build_formatted_doc
from aiguru.phase2.retriever import expand_legal_query
from aiguru.phase2.retriever import ScoredChunk
from aiguru.phase2.bm25_indexer_simple import simple_tokenize
from aiguru.phase2.cache import RetrievalCache
from aiguru.phase4.postprocess import PostProcessor
from aiguru.submission import validate_and_package, validate_results

from test_phase3_4 import chunk


class PipelineQualityTests(unittest.TestCase):
    def test_phapdien_source_note_becomes_submission_article(self):
        row = {
            "article_title": "Điều 1.1.LQ.1. Phạm vi điều chỉnh",
            "subject_title": "Hỗ trợ doanh nghiệp nhỏ và vừa",
            "source_note_text": "Điều 4 Luật số 04/2017/QH14 về hỗ trợ doanh nghiệp nhỏ và vừa",
            "content_text": "Doanh nghiệp nhỏ và vừa bao gồm doanh nghiệp siêu nhỏ, nhỏ và vừa.",
        }
        document = normalize_legal_doc(row, "phapdien")
        chunks = chunk_legal_doc(document)
        self.assertEqual(document["doc_id"], "04/2017/QH14")
        self.assertEqual(chunks[0].metadata.article_number, "Điều 4")
        self.assertTrue(chunks[0].metadata.submission_eligible)

    def test_internal_phapdien_article_is_not_submitted(self):
        document = normalize_legal_doc(
            {
                "article_title": "Điều 1.1.LQ.1. Phạm vi điều chỉnh",
                "source_links": [{"text": "Luật số 04/2017/QH14 về hỗ trợ DNNVV"}],
                "content_text": "Nội dung pháp điển không nêu rõ Điều gốc của văn bản.",
            },
            "phapdien",
        )
        self.assertFalse(chunk_legal_doc(document)[0].metadata.submission_eligible)

    def test_multiple_articles_from_one_source_document_are_preserved(self):
        document = normalize_legal_doc(
            {
                "article_title": "Điều pháp điển",
                "source_note_text": "Điều 4 và Điều 5 Luật số 04/2017/QH14 về hỗ trợ DNNVV",
                "content_text": "Nội dung pháp lý liên quan đến nhiều điều của cùng một văn bản.",
            },
            "phapdien",
        )
        self.assertEqual(
            [chunk.metadata.article_number for chunk in chunk_legal_doc(document)],
            ["Điều 4", "Điều 5"],
        )

    def test_normalize_doc_id_rejects_internal_anchor(self):
        self.assertEqual(normalize_doc_id("#0100100000000000100000100000000000000000"), "")

    def test_formatted_doc_enforces_competition_prefix(self):
        value = build_formatted_doc(
            "04/2017/QH14",
            "Luật",
            "Luật số 04/2017/QH14 về hỗ trợ doanh nghiệp nhỏ và vừa",
        )
        self.assertEqual(value, "04/2017/QH14|Luật 04/2017/QH14 về hỗ trợ doanh nghiệp nhỏ và vừa")

    def test_real_phapdien_note_strips_issue_date_and_issuer(self):
        document = normalize_legal_doc(
            {
                "article_title": "Điều 1.1.LQ.1. Phạm vi điều chỉnh",
                "source_note_text": (
                    "(Điều 1 Luật số 32/2004/QH11 An ninh Quốc gia ngày "
                    "03/12/2004 của Quốc hội, có hiệu lực thi hành kể từ ngày 01/07/2005 )"
                ),
                "content_text": "Luật này quy định về chính sách an ninh quốc gia.",
            },
            "phapdien",
        )
        formatted = chunk_legal_doc(document)[0].metadata.formatted_doc
        self.assertEqual(formatted, "32/2004/QH11|Luật 32/2004/QH11 An ninh Quốc gia")

    def test_query_expansion_adds_legal_domain(self):
        expanded = expand_legal_query("Công ty chậm đóng bảo hiểm xã hội thì sao?")
        self.assertIn("Luật Bảo hiểm xã hội", expanded)

    def test_legal_tokenizer_adds_curated_phrase_without_ngram_explosion(self):
        tokens = simple_tokenize("Doanh nghiệp nhỏ và vừa được hỗ trợ.")
        self.assertIn("doanh_nghiệp_nhỏ_và_vừa", tokens)
        self.assertLess(len(tokens), 10)

    def test_document_titles_are_canonicalized(self):
        row = {
            "article_title": "Điều 1",
            "source_note_text": "Điều 4 Luật số 04/2017/QH14 về hỗ trợ DNNVV",
            "content_text": "Nội dung đủ dài để tạo chunk căn cứ pháp lý cho doanh nghiệp nhỏ và vừa.",
        }
        chunks = chunk_legal_doc(normalize_legal_doc(row, "phapdien"))
        normalized = canonicalize_document_titles(chunks)
        self.assertEqual(normalized[0].metadata.formatted_doc, chunks[0].metadata.formatted_doc)

    def test_postprocessor_removes_unsupported_article(self):
        result = PostProcessor().build_result(
            1,
            "Câu hỏi",
            "Theo Điều 99, doanh nghiệp phải thực hiện nghĩa vụ.",
            [chunk("Điều 4")],
        )
        self.assertNotIn("Điều 99", result["answer"])
        self.assertIn("Điều 4", result["answer"])

    def test_submission_validator_and_flat_zip(self):
        questions = [{"id": 1, "question": "Q"}]
        results = [{
            "id": 1,
            "question": "Q",
            "answer": "Theo Điều 4.",
            "relevant_docs": ["04/2017/QH14|Luật 04/2017/QH14 Hỗ trợ doanh nghiệp nhỏ và vừa"],
            "relevant_articles": ["04/2017/QH14|Luật 04/2017/QH14 Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 4"],
        }]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            results_path = directory / "results.json"
            questions_path = directory / "questions.json"
            results_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
            questions_path.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")
            output = validate_and_package(results_path, questions_path, directory / "submission.zip")
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(archive.namelist(), ["results.json"])

    def test_submission_validator_rejects_article_missing_from_answer(self):
        errors = validate_results(
            [{
                "id": 1,
                "question": "Q",
                "answer": "Theo Điều 4.",
                "relevant_docs": ["04/2017/QH14|Luật 04/2017/QH14 SME"],
                "relevant_articles": [
                    "04/2017/QH14|Luật 04/2017/QH14 SME|Điều 4",
                    "04/2017/QH14|Luật 04/2017/QH14 SME|Điều 5",
                ],
            }],
            [{"id": 1, "question": "Q"}],
        )
        self.assertTrue(any("missing from answer citations" in error for error in errors))

    def test_retrieval_cache_resumes_after_partial_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retrieval.jsonl"
            path.write_text('{"id": 1, "question": "unfinished"', encoding="utf-8")
            cache = RetrievalCache(path)
            scored = ScoredChunk(
                node={"chunk_id": "chunk-1", "text": "Nội dung", "metadata": {}},
                score=0.8,
            )
            self.assertTrue(cache.append(1, "Câu hỏi", [scored]))
            resumed = RetrievalCache(path)
            self.assertEqual(resumed.completed_ids, {1})
            self.assertEqual(resumed.retrieve_by_id(1)[0]["chunk_id"], "chunk-1")

    def test_macro_f2(self):
        article = "04/2017/QH14|Luật 04/2017/QH14 SME|Điều 4"
        metrics = macro_retrieval_metrics(
            [{"id": 1, "relevant_articles": [article]}],
            [{"id": 1, "relevant_articles": [article]}],
        )
        self.assertEqual(metrics["macro_f2"], 1.0)

    def test_silver_retrieval_recall(self):
        metrics = silver_retrieval_recall(
            [{"id": 1, "question": "Theo Điều 4 Luật thì xử lý thế nào?"}],
            [{"id": 1, "chunks": [{
                "metadata": {
                    "article_number": "Điều 4",
                    "doc_id": "04/2017/QH14",
                    "submission_eligible": True,
                }
            }]}],
        )
        self.assertEqual(metrics["silver_recall_at_3"], 1.0)


if __name__ == "__main__":
    unittest.main()
