"""
Benchmark Runner Module.

Executes end-to-end evaluation runs against golden datasets:
retrieval top-K scoring, answer generation, metric calculation, and comparison sweeps.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Sequence
from dataclasses import dataclass, field

from src.evaluation.dataset_loader import BenchmarkQuestion
from src.evaluation.retrieval_metrics import evaluate_single_query, QueryRetrievalResult
from src.evaluation.answer_metrics import AnswerMetricsEvaluator, AnswerQualityScore
from src.retrieval.retriever import Retriever
from src.generation.answer_generator import AnswerGenerator


@dataclass
class QuestionBenchmarkResult:
    """Individual question evaluation outcome across retrieval and generation."""
    question_id: str
    question: str
    reference_answer: str
    ground_truth_chunk_ids: List[str]
    retrieved_chunk_ids: List[str]
    hit_at_k: int
    recall_at_k: float
    mrr: float
    precision_at_k: float
    generated_answer: str
    faithfulness: Optional[float]
    answer_relevance: Optional[float]
    answer_correctness: Optional[float]
    latency_ms: float
    citations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary matching specification table format."""
        return {
            "Question": self.question_id,
            "Query": self.question,
            "Reference": self.reference_answer,
            "Ground_Truth_IDs": self.ground_truth_chunk_ids,
            "Retrieved_IDs": self.retrieved_chunk_ids,
            "Hit@K": self.hit_at_k,
            "Recall@K": round(self.recall_at_k, 4),
            "MRR": round(self.mrr, 4),
            "Precision@K": round(self.precision_at_k, 4),
            "Faithfulness": str(round(self.faithfulness, 2)) if self.faithfulness is not None else "N/A",
            "Relevance": str(round(self.answer_relevance, 2)) if self.answer_relevance is not None else "N/A",
            "Correctness": str(round(self.answer_correctness, 2)) if self.answer_correctness is not None else "N/A",
            "Generated_Answer": self.generated_answer,
            "Latency_ms": round(self.latency_ms, 2)
        }


@dataclass
class BenchmarkSummary:
    """Consolidated benchmark run summary."""
    experiment_id: str
    timestamp: str
    k: int
    num_questions: int
    hit_rate_at_k: float
    recall_at_k: float
    mrr: float
    precision_at_k: float
    mean_faithfulness: Optional[float]
    mean_relevance: Optional[float]
    mean_correctness: Optional[float]
    avg_latency_ms: float
    per_question_results: List[QuestionBenchmarkResult]

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "k": self.k,
            "num_questions": self.num_questions,
            "hit_rate_at_k": round(self.hit_rate_at_k, 4),
            "recall_at_k": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            "precision_at_k": round(self.precision_at_k, 4),
            "mean_faithfulness": round(self.mean_faithfulness, 4) if self.mean_faithfulness is not None else "N/A",
            "mean_relevance": round(self.mean_relevance, 4) if self.mean_relevance is not None else "N/A",
            "mean_correctness": round(self.mean_correctness, 4) if self.mean_correctness is not None else "N/A",
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "per_question_results": [r.to_dict() for r in self.per_question_results]
        }


class BenchmarkRunner:
    """
    Coordinates end-to-end evaluation against golden datasets.
    """

    def __init__(
        self,
        retriever: Retriever,
        answer_generator: Optional[AnswerGenerator] = None,
        answer_evaluator: Optional[AnswerMetricsEvaluator] = None
    ):
        self.retriever = retriever
        self.answer_generator = answer_generator or AnswerGenerator()
        self.answer_evaluator = answer_evaluator or AnswerMetricsEvaluator()

    def run_benchmark(
        self,
        questions: List[BenchmarkQuestion],
        k: int = 5,
        generate_answers: bool = True,
        experiment_id: Optional[str] = None
    ) -> BenchmarkSummary:
        """
        Execute full benchmark across the provided questions for rank K.

        Args:
            questions: List of BenchmarkQuestion objects.
            k: Top-K cutoff.
            generate_answers: Whether to run answer generation and quality evaluation.
            experiment_id: Optional tracking identifier.

        Returns:
            BenchmarkSummary containing aggregate and granular metrics.
        """
        exp_id = experiment_id or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        timestamp = datetime.now().isoformat()

        if not questions:
            return BenchmarkSummary(
                experiment_id=exp_id,
                timestamp=timestamp,
                k=k,
                num_questions=0,
                hit_rate_at_k=0.0,
                recall_at_k=0.0,
                mrr=0.0,
                precision_at_k=0.0,
                mean_faithfulness=None,
                mean_relevance=None,
                mean_correctness=None,
                avg_latency_ms=0.0,
                per_question_results=[]
            )

        per_question_results: List[QuestionBenchmarkResult] = []
        total_hits = 0
        total_recall = 0.0
        total_rr = 0.0
        total_precision = 0.0
        faithfulness_scores: List[float] = []
        relevance_scores: List[float] = []
        correctness_scores: List[float] = []
        latencies: List[float] = []

        for bq in questions:
            q_start = time.perf_counter()

            # 1. Retrieve candidates
            candidates = self.retriever.retrieve(query=bq.question, top_k=k)
            retrieved_ids = [c.chunk_id for c in candidates]

            # 2. Evaluate retrieval deterministically
            ret_eval = evaluate_single_query(
                question_id=bq.question_id,
                retrieved_chunk_ids=retrieved_ids,
                ground_truth_chunk_ids=bq.relevant_chunk_ids,
                k=k
            )

            total_hits += int(ret_eval.hit)
            total_recall += ret_eval.recall
            total_rr += ret_eval.reciprocal_rank
            total_precision += ret_eval.precision

            # 3. Generation and Answer Evaluation
            answer_text = ""
            citations_list: List[Dict[str, Any]] = []
            faithfulness_val: Optional[float] = None
            relevance_val: Optional[float] = None
            correctness_val: Optional[float] = None

            if generate_answers:
                gen_result = self.answer_generator.generate_answer(
                    question=bq.question,
                    retrieved_chunks=candidates
                )
                answer_text = gen_result.answer
                citations_list = [
                    {
                        "document": c.document_name,
                        "page": c.page_number,
                        "chunk_id": c.chunk_id,
                        "snippet": c.snippet
                    }
                    for c in gen_result.citations
                ]

                context_strings = [c.text for c in candidates]
                ans_eval = self.answer_evaluator.evaluate_single(
                    question_id=bq.question_id,
                    question=bq.question,
                    answer=answer_text,
                    retrieved_contexts=context_strings,
                    reference_answer=bq.reference_answer,
                    retrieval_hit=ret_eval.hit
                )

                faithfulness_val = ans_eval.faithfulness
                relevance_val = ans_eval.answer_relevance
                correctness_val = ans_eval.answer_correctness

                if faithfulness_val is not None:
                    faithfulness_scores.append(faithfulness_val)
                if relevance_val is not None:
                    relevance_scores.append(relevance_val)
                if correctness_val is not None:
                    correctness_scores.append(correctness_val)

            elapsed_ms = (time.perf_counter() - q_start) * 1000.0
            latencies.append(elapsed_ms)

            per_question_results.append(
                QuestionBenchmarkResult(
                    question_id=bq.question_id,
                    question=bq.question,
                    reference_answer=bq.reference_answer,
                    ground_truth_chunk_ids=bq.relevant_chunk_ids,
                    retrieved_chunk_ids=retrieved_ids,
                    hit_at_k=int(ret_eval.hit),
                    recall_at_k=ret_eval.recall,
                    mrr=ret_eval.reciprocal_rank,
                    precision_at_k=ret_eval.precision,
                    generated_answer=answer_text,
                    faithfulness=faithfulness_val,
                    answer_relevance=relevance_val,
                    answer_correctness=correctness_val,
                    latency_ms=elapsed_ms,
                    citations=citations_list
                )
            )

        n = len(questions)
        mean_faith = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None
        mean_rel = sum(relevance_scores) / len(relevance_scores) if relevance_scores else None
        mean_corr = sum(correctness_scores) / len(correctness_scores) if correctness_scores else None
        avg_lat = sum(latencies) / n if n > 0 else 0.0

        return BenchmarkSummary(
            experiment_id=exp_id,
            timestamp=timestamp,
            k=k,
            num_questions=n,
            hit_rate_at_k=total_hits / n,
            recall_at_k=total_recall / n,
            mrr=total_rr / n,
            precision_at_k=total_precision / n,
            mean_faithfulness=mean_faith,
            mean_relevance=mean_rel,
            mean_correctness=mean_corr,
            avg_latency_ms=avg_lat,
            per_question_results=per_question_results
        )

    def run_k_sweep(
        self,
        questions: List[BenchmarkQuestion],
        k_values: Sequence[int] = (1, 3, 5),
        generate_answers: bool = False
    ) -> Dict[int, BenchmarkSummary]:
        """
        Compare retrieval performance across varying K thresholds (e.g. 1, 3, 5).
        """
        results: Dict[int, BenchmarkSummary] = {}
        for k_val in k_values:
            results[k_val] = self.run_benchmark(
                questions=questions,
                k=k_val,
                generate_answers=generate_answers,
                experiment_id=f"sweep_k{k_val}"
            )
        return results
