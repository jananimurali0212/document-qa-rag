"""
FastAPI Server for Document Q&A RAG Studio & Evaluation Harness.

Provides high-performance REST APIs and serves the modern Single-Page Application (SPA) UI:
- Document ingestion (PDF, DOCX, TXT) with chunking & embedding
- Grounded Q&A with verifiable citations and Gemini 3.8 Flash synthesis
- Retrieval and answer quality benchmarking against golden datasets
- Real-time parameter sweeps and telemetry
"""

from __future__ import annotations

import os
import sys
import io
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Ensure utf-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query as QueryParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Internal RAG modules
from src.ingestion.file_validator import FileValidator, IngestionStatus
from src.ingestion.document_parser import DocumentParser, ParsingError
from src.ingestion.chunker import Chunker, ChunkRecord
from src.embeddings.embedding_service import EmbeddingService
from src.retrieval.vector_store import VectorStore, SearchResult
from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker
from src.generation.prompt_builder import PromptBuilder
from src.generation.answer_generator import AnswerGenerator
from src.evaluation.dataset_loader import DatasetLoader, BenchmarkQuestion
from src.evaluation.retrieval_metrics import evaluate_single_query
from src.evaluation.answer_metrics import AnswerMetricsEvaluator
from src.evaluation.benchmark_runner import BenchmarkRunner, BenchmarkSummary
from src.evaluation.report_generator import ReportGenerator
from src.storage.metadata_store import MetadataStore


# =====================================================================
# APPLICATION & PIPELINE STATE
# =====================================================================

app = FastAPI(
    title="Document Q&A RAG Studio",
    description="Production-grade RAG and Evaluation Platform",
    version="2.0.0"
)

# Enable CORS for local client development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RAGPipelineManager:
    """Manages persistent singletons for vector storage, embeddings, and retrieval."""

    def __init__(self):
        self.metadata_store = MetadataStore(db_path="rag_metadata.db")
        self.embedding_service = EmbeddingService(
            model_name="all-MiniLM-L6-v2",
            use_fallback_if_unavailable=True
        )
        self.vector_store = VectorStore(
            dimension=self.embedding_service.dimension,
            use_faiss=True
        )
        self.reranker = Reranker(use_neural=False)
        self.file_validator = FileValidator()
        self.document_parser = DocumentParser()

        # Cache for latest benchmark result
        self.latest_benchmark: Optional[BenchmarkSummary] = None
        self.latest_k_sweep: Optional[Dict[int, BenchmarkSummary]] = None

        # Synchronize SQLite database chunks into VectorStore on startup
        self._sync_vector_store()

    def _sync_vector_store(self):
        """Restore existing chunks from SQLite into vector store memory."""
        try:
            chunks = self.metadata_store.get_all_chunks()
            if chunks:
                print(f"[RAG Init] Loading {len(chunks)} existing chunks from SQLite...")
                embeddings = self.embedding_service.embed_chunks(chunks)
                self.vector_store.add_chunks(chunks, embeddings)
                print(f"[RAG Init] Successfully synchronized {len(chunks)} chunks into vector store.")
        except Exception as e:
            print(f"[RAG Init Warning] Could not preload chunks from SQLite: {e}")

    def get_api_key(self) -> str:
        return os.getenv("GEMINI_API_KEY", "").strip()

    def get_retriever(self, enable_reranker: bool = True) -> Retriever:
        return Retriever(
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
            reranker=self.reranker if enable_reranker else None
        )

    def get_answer_generator(self) -> AnswerGenerator:
        return AnswerGenerator(
            api_key=self.get_api_key(),
            model_name="gemini-3.8-flash",
            prompt_builder=PromptBuilder()
        )


pipeline = RAGPipelineManager()


# =====================================================================
# REQUEST & RESPONSE MODELS
# =====================================================================

class QueryRequest(BaseModel):
    question: str = Field(..., description="The query string to answer.")
    top_k: int = Field(5, ge=1, le=10, description="Number of retrieved context chunks.")
    enable_reranker: bool = Field(True, description="Whether to apply neural/lexical reranking.")


class BenchmarkRequest(BaseModel):
    k: int = Field(5, ge=1, le=10, description="K parameter for retrieval metrics.")
    dataset_path: str = Field("evaluation_data/evaluation_dataset.json", description="Path to evaluation dataset.")
    include_sweep: bool = Field(True, description="Run K in [1, 3, 5] comparative sweep.")


# =====================================================================
# REST ENDPOINTS
# =====================================================================

@app.get("/api/status")
async def get_system_status():
    """Return health, connectivity, and telemetry metrics."""
    api_key = pipeline.get_api_key()
    stats = pipeline.vector_store.get_stats()
    docs = pipeline.metadata_store.get_all_documents()

    return {
        "status": "online",
        "llm_connected": bool(api_key),
        "model_name": "gemini-3.8-flash",
        "embedder": "Deterministic Fallback" if pipeline.embedding_service.is_fallback else "SentenceTransformers (all-MiniLM-L6-v2)",
        "vector_backend": stats.get("backend", "NumPy Cosine"),
        "total_documents": len(docs),
        "total_chunks": stats.get("total_chunks", 0),
        "dimension": pipeline.embedding_service.dimension
    }


@app.get("/api/documents")
async def list_documents():
    """List all ingested documents with page counts and metadata."""
    docs = pipeline.metadata_store.get_all_documents()
    return {"documents": docs, "count": len(docs)}


@app.get("/api/documents/{document_id}/chunks")
async def get_document_chunks(document_id: str):
    """Retrieve chunk details for a given document."""
    chunks = pipeline.metadata_store.get_chunks_for_document(document_id)
    return {"document_id": document_id, "chunks": chunks, "count": len(chunks)}


@app.post("/api/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(100)
):
    """
    Ingest uploaded PDF, DOCX, or TXT documents.
    Validates, parses, chunks, embeds, and updates both VectorStore and SQLite.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    current_docs = pipeline.metadata_store.get_all_documents()
    current_count = len(current_docs)

    results = []
    total_new_chunks = 0

    for idx, upload in enumerate(files):
        file_bytes = await upload.read()
        doc_counter = current_count + idx + 1

        # 1. Validation
        val_res = pipeline.file_validator.validate_file(
            file_bytes=file_bytes,
            filename=upload.filename or f"doc_{doc_counter}",
            sequential_counter=doc_counter
        )

        if not val_res.is_valid:
            results.append({
                "filename": upload.filename,
                "status": "error",
                "message": val_res.error_message
            })
            continue

        # 2. Parsing
        try:
            parsed_doc = pipeline.document_parser.parse(
                file_bytes=file_bytes,
                document_id=val_res.document_id,
                document_name=val_res.document_name
            )
        except Exception as e:
            results.append({
                "filename": upload.filename,
                "status": "error",
                "message": f"Parsing failed: {str(e)}"
            })
            continue

        # 3. Chunking
        chunks = chunker.chunk_document(parsed_doc)
        if not chunks:
            results.append({
                "filename": upload.filename,
                "status": "warning",
                "message": "No extractable text content found in document."
            })
            continue

        # 4. Embeddings & Vector Indexing
        embeddings = pipeline.embedding_service.embed_chunks(chunks)
        pipeline.vector_store.add_chunks(chunks, embeddings)

        # 5. SQLite Persistence
        pipeline.metadata_store.record_document(
            document_id=val_res.document_id,
            document_name=val_res.document_name,
            file_type=val_res.file_type,
            file_size_bytes=val_res.file_size_bytes,
            num_pages=parsed_doc.total_pages,
            num_chunks=len(chunks),
            status=IngestionStatus.INDEXED
        )
        pipeline.metadata_store.record_chunks(chunks)

        total_new_chunks += len(chunks)
        results.append({
            "document_id": val_res.document_id,
            "filename": val_res.document_name,
            "status": "success",
            "pages": parsed_doc.total_pages,
            "chunks": len(chunks),
            "size_bytes": val_res.file_size_bytes
        })

    return {
        "message": f"Processed {len(files)} file(s). Added {total_new_chunks} text chunk(s).",
        "results": results,
        "total_documents": len(pipeline.metadata_store.get_all_documents()),
        "total_chunks": pipeline.vector_store.get_stats().get("total_chunks", 0)
    }


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Purge document and its chunk embeddings."""
    deleted_vec = pipeline.vector_store.delete_document(document_id)
    deleted_meta = pipeline.metadata_store.delete_document(document_id)

    if not deleted_meta and not deleted_vec:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    return {
        "message": f"Document '{document_id}' successfully removed.",
        "document_id": document_id
    }


@app.post("/api/documents/clear")
async def clear_all_documents():
    """Clear all vector embeddings and SQLite document records."""
    pipeline.vector_store.clear()
    pipeline.metadata_store.clear_all()
    return {"message": "Knowledge base completely cleared."}


@app.post("/api/query")
async def run_query(payload: QueryRequest):
    """
    Execute semantic retrieval and generative or grounded extractive Q&A.
    Returns generated answer with verifiable citations and audit telemetry.
    """
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    stats = pipeline.vector_store.get_stats()
    if stats.get("total_chunks", 0) == 0:
        return {
            "question": question,
            "answer": "The knowledge base is currently empty. Please upload documents in the Knowledge Base tab first.",
            "citations": [],
            "latency_ms": 0.0,
            "is_llm": False,
            "retrieved_chunks_count": 0
        }

    start_time = time.perf_counter()

    # 1. Retrieval
    retriever = pipeline.get_retriever(enable_reranker=payload.enable_reranker)
    search_results = retriever.retrieve(query=question, top_k=payload.top_k)

    # 2. Generation / Extraction
    generator = pipeline.get_answer_generator()
    gen_result = generator.generate_answer(
        question=question,
        retrieved_chunks=search_results
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # Format citations
    citations_data = [
        {
            "chunk_id": c.chunk_id,
            "document_name": c.document_name,
            "page_number": c.page_number,
            "score": round(c.score, 4),
            "snippet": c.snippet
        }
        for c in gen_result.citations
    ]

    return {
        "question": question,
        "answer": gen_result.answer,
        "citations": citations_data,
        "latency_ms": round(elapsed_ms, 2),
        "is_llm": bool(pipeline.get_api_key()),
        "model_name": gen_result.model_name,
        "retrieved_chunks_count": len(search_results),
        "grounded": gen_result.grounded
    }


@app.post("/api/benchmark/run")
async def run_benchmark(payload: BenchmarkRequest):
    """
    Execute end-to-end benchmark against golden evaluation dataset.
    Computes Hit Rate@K, Recall@K, MRR, Precision@K, Faithfulness, and Latency.
    """
    dataset_file = Path(payload.dataset_path)
    if not dataset_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Evaluation dataset file not found at '{payload.dataset_path}'"
        )

    try:
        questions = DatasetLoader.load_from_file(dataset_file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load dataset: {e}")

    retriever = pipeline.get_retriever(enable_reranker=True)
    generator = pipeline.get_answer_generator()
    evaluator = AnswerMetricsEvaluator()

    runner = BenchmarkRunner(
        retriever=retriever,
        answer_generator=generator,
        answer_evaluator=evaluator
    )

    # Run primary evaluation
    summary = runner.run_benchmark(
        questions=questions,
        k=payload.k,
        generate_answers=True
    )
    pipeline.latest_benchmark = summary

    # Persist summary in SQLite
    pipeline.metadata_store.record_experiment(
        experiment_id=summary.experiment_id,
        k=summary.k,
        num_questions=summary.num_questions,
        hit_rate_at_k=summary.hit_rate_at_k,
        recall_at_k=summary.recall_at_k,
        mrr=summary.mrr,
        precision_at_k=summary.precision_at_k,
        faithfulness=summary.mean_faithfulness,
        avg_latency_ms=summary.avg_latency_ms,
        summary_dict=summary.to_dict()
    )

    # Optional K-sweep for chart visualization (K = 1, 3, 5)
    k_sweep_data = {}
    if payload.include_sweep:
        sweep_summaries = runner.run_k_sweep(
            questions=questions,
            k_values=(1, 3, 5),
            generate_answers=False
        )
        pipeline.latest_k_sweep = sweep_summaries
        for k_val, s in sweep_summaries.items():
            k_sweep_data[k_val] = {
                "k": k_val,
                "hit_rate": round(s.hit_rate_at_k, 4),
                "recall": round(s.recall_at_k, 4),
                "mrr": round(s.mrr, 4),
                "precision": round(s.precision_at_k, 4)
            }

    # Per-question breakdown
    per_question = [q.to_dict() for q in summary.per_question_results]

    return {
        "summary": summary.to_dict(),
        "k_sweep": k_sweep_data,
        "per_question": per_question
    }


@app.get("/api/benchmark/history")
async def get_benchmark_history():
    """Retrieve historical benchmark experiment records."""
    experiments = pipeline.metadata_store.get_experiments()
    return {"experiments": experiments, "count": len(experiments)}


@app.get("/api/benchmark/dataset")
async def get_benchmark_dataset():
    """Retrieve golden evaluation dataset questions overview."""
    dataset_file = Path("evaluation_data/evaluation_dataset.json")
    if not dataset_file.exists():
        return {"questions": [], "count": 0}

    try:
        questions = DatasetLoader.load_from_file(dataset_file)
    except Exception:
        return {"questions": [], "count": 0}
    return {
        "count": len(questions),
        "questions": [
            {
                "question_id": q.question_id,
                "question": q.question,
                "reference_answer": q.reference_answer,
                "ground_truth_chunks": q.relevant_chunk_ids
            }
            for q in questions
        ]
    }


@app.get("/api/benchmark/export/{export_format}")
async def export_benchmark_results(export_format: str):
    """Export the most recent benchmark run as CSV or JSON."""
    if not pipeline.latest_benchmark:
        raise HTTPException(
            status_code=400,
            detail="No benchmark run found to export. Please run a benchmark first."
        )

    summary = pipeline.latest_benchmark
    rows = [q.to_dict() for q in summary.per_question_results]

    if export_format.lower() == "json":
        data = {
            "summary": summary.to_dict(),
            "results": rows
        }
        return Response(
            content=json.dumps(data, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=rag_benchmark_{summary.experiment_id}.json"}
        )
    elif export_format.lower() == "csv":
        import csv
        output = io.StringIO()
        if rows:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                clean_row = {}
                for k, v in r.items():
                    if isinstance(v, list):
                        clean_row[k] = "; ".join(map(str, v))
                    else:
                        clean_row[k] = v
                writer.writerow(clean_row)

        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=rag_benchmark_{summary.experiment_id}.csv"}
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid export format. Choose 'csv' or 'json'.")


# =====================================================================
# STATIC ASSETS MOUNTING
# =====================================================================

# Ensure web directory exists
web_dir = Path("web")
web_dir.mkdir(exist_ok=True)

# Mount static web directory
app.mount("/", StaticFiles(directory="web", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("[INFO] Starting Document Q&A RAG Studio Web Server")
    print("[INFO] Open http://localhost:8000 in your web browser")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
