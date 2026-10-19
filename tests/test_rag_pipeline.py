"""
Integration Tests for End-to-End RAG Pipeline and Evaluation Harness.

Validates the full workflow:
Parsing -> Chunking -> Vector Indexing -> Retrieval -> Reranking -> Answer Generation -> Evaluation -> Reports.
"""

import pytest
from pathlib import Path

from src.ingestion.file_validator import FileValidator
from src.ingestion.document_parser import DocumentParser
from src.ingestion.chunker import Chunker, ChunkRecord
from src.embeddings.embedding_service import EmbeddingService
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker
from src.generation.prompt_builder import PromptBuilder
from src.generation.answer_generator import AnswerGenerator
from src.evaluation.dataset_loader import BenchmarkQuestion
from src.evaluation.benchmark_runner import BenchmarkRunner
from src.evaluation.report_generator import ReportGenerator
from src.storage.metadata_store import MetadataStore


@pytest.fixture
def rag_pipeline():
    """Build a complete local RAG pipeline instance."""
    embed_service = EmbeddingService(dimension=384, use_fallback_if_unavailable=True)
    vector_store = VectorStore(dimension=embed_service.dimension, use_faiss=False)
    reranker = Reranker(use_neural=False)
    retriever = Retriever(embedding_service=embed_service, vector_store=vector_store, reranker=reranker)
    prompt_builder = PromptBuilder()
    answer_gen = AnswerGenerator(prompt_builder=prompt_builder)
    return {
        "embed_service": embed_service,
        "vector_store": vector_store,
        "retriever": retriever,
        "answer_gen": answer_gen,
    }


class TestRAGPipelineEndToEnd:
    """Full lifecycle testing from raw text to evaluation reports."""

    def test_ingestion_indexing_and_retrieval(self, rag_pipeline):
        doc_parser = DocumentParser()
        chunker = Chunker(chunk_size=120, chunk_overlap=30)
        vector_store = rag_pipeline["vector_store"]
        retriever = rag_pipeline["retriever"]
        embed_service = rag_pipeline["embed_service"]

        sample_text = (
            "The Document Q&A Chatbot enables users to upload PDF, DOCX, and TXT files. "
            "It uses Retrieval-Augmented Generation to answer questions with citations. "
            "An evaluation harness calculates Hit Rate@K, Recall@K, MRR, and Precision@K."
        )

        parsed = doc_parser.parse_txt(sample_text.encode("utf-8"), document_id="doc01", document_name="system_spec.txt")
        chunks = chunker.chunk_document(parsed)

        assert len(chunks) >= 2
        assert chunks[0].chunk_id == "doc01_chunk_000"
        assert chunks[0].document_name == "system_spec.txt"

        # Index chunks
        embeddings = embed_service.embed_chunks(chunks)
        vector_store.add_chunks(chunks, embeddings)

        stats = vector_store.get_stats()
        assert stats["total_chunks"] == len(chunks)
        assert stats["total_documents"] == 1

        # Retrieve top 2
        results = retriever.retrieve("What metrics does the evaluation harness calculate?", top_k=2)
        assert len(results) > 0
        assert results[0].document_id == "doc01"
        assert results[0].score > 0.0
        assert results[0].chunk_id.startswith("doc01_chunk_")

    def test_answer_generation_and_citations(self, rag_pipeline):
        retriever = rag_pipeline["retriever"]
        answer_gen = rag_pipeline["answer_gen"]

        # Ensure index has content from previous test or add fresh chunk
        chunk = ChunkRecord(
            chunk_id="doc01_chunk_004",
            document_id="doc01",
            document_name="guide.pdf",
            page_number=5,
            text="The evaluation harness calculates Hit Rate@K, Recall@K, MRR, and Precision@K.",
            chunk_index=4
        )
        rag_pipeline["vector_store"].add_chunks(
            [chunk],
            rag_pipeline["embed_service"].embed_chunks([chunk])
        )

        candidates = retriever.retrieve("What metrics are calculated?", top_k=1)
        assert len(candidates) == 1

        ans_result = answer_gen.generate_answer("What metrics are calculated?", candidates)
        assert len(ans_result.answer) > 10
        assert ans_result.grounded is True
        assert len(ans_result.citations) >= 1
        assert ans_result.citations[0].document_name == "guide.pdf"
        assert ans_result.citations[0].page_number == 5

    def test_abstain_when_context_is_empty(self, rag_pipeline):
        answer_gen = rag_pipeline["answer_gen"]
        # Empty context
        ans_result = answer_gen.generate_answer("What is quantum entanglement?", [])
        assert ans_result.abstained is True
        assert "sufficient information" in ans_result.answer.lower() or "insufficient" in ans_result.answer.lower()

    def test_benchmark_runner_and_report_export(self, rag_pipeline, tmp_path):
        retriever = rag_pipeline["retriever"]
        answer_gen = rag_pipeline["answer_gen"]

        # Index sample chunk
        chunk = ChunkRecord(
            chunk_id="doc01_chunk_002",
            document_id="doc01",
            document_name="architecture.pdf",
            page_number=2,
            text="The system architecture separates ingestion, retrieval, and generation.",
            chunk_index=2
        )
        rag_pipeline["vector_store"].add_chunks(
            [chunk],
            rag_pipeline["embed_service"].embed_chunks([chunk])
        )

        benchmark_questions = [
            BenchmarkQuestion(
                question_id="Q001",
                question="How is the system architecture structured?",
                reference_answer="It separates ingestion, retrieval, and generation.",
                relevant_chunk_ids=["doc01_chunk_002"]
            )
        ]

        runner = BenchmarkRunner(retriever=retriever, answer_generator=answer_gen)
        summary = runner.run_benchmark(benchmark_questions, k=1, generate_answers=True)

        assert summary.num_questions == 1
        assert summary.hit_rate_at_k == 1.0
        assert summary.recall_at_k == 1.0
        assert summary.mrr == 1.0
        assert summary.precision_at_k == 1.0
        assert len(summary.per_question_results) == 1

        # Test report exports
        csv_file = tmp_path / "benchmark_test.csv"
        json_file = tmp_path / "benchmark_test.json"

        csv_content = ReportGenerator.export_csv(summary, output_path=str(csv_file))
        json_content = ReportGenerator.export_json(summary, output_path=str(json_file))
        md_report = ReportGenerator.generate_markdown_report(summary)

        assert csv_file.exists()
        assert json_file.exists()
        assert "Q001" in csv_content
        assert "Q001" in json_content
        assert "Document QA RAG — Evaluation Benchmark Report" in md_report

    def test_document_lifecycle_deletion(self, rag_pipeline):
        vector_store = rag_pipeline["vector_store"]
        chunk1 = ChunkRecord("docA_001", "docA", "fileA.txt", 1, "Alpha text", 0)
        chunk2 = ChunkRecord("docB_001", "docB", "fileB.txt", 1, "Beta text", 0)

        embs = rag_pipeline["embed_service"].embed_chunks([chunk1, chunk2])
        vector_store.add_chunks([chunk1, chunk2], embs)

        assert vector_store.get_stats()["total_chunks"] >= 2
        deleted = vector_store.delete_document("docA")
        assert deleted == 1
        assert vector_store.get_chunk_by_id("docA_001") is None
        assert vector_store.get_chunk_by_id("docB_001") is not None
