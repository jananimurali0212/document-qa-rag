"""Evaluation module: golden dataset loader, deterministic retrieval metrics, and benchmark runner."""
from src.evaluation.dataset_loader import DatasetLoader, BenchmarkQuestion
from src.evaluation.retrieval_metrics import (
    RetrievalMetrics,
    QueryRetrievalResult,
    calculate_hit_rate_at_k,
    calculate_recall_at_k,
    calculate_mrr,
    calculate_precision_at_k,
)
from src.evaluation.answer_metrics import AnswerMetricsEvaluator, AnswerQualityScore
from src.evaluation.benchmark_runner import BenchmarkRunner, BenchmarkSummary, QuestionBenchmarkResult
from src.evaluation.report_generator import ReportGenerator

__all__ = [
    "DatasetLoader",
    "BenchmarkQuestion",
    "RetrievalMetrics",
    "QueryRetrievalResult",
    "calculate_hit_rate_at_k",
    "calculate_recall_at_k",
    "calculate_mrr",
    "calculate_precision_at_k",
    "AnswerMetricsEvaluator",
    "AnswerQualityScore",
    "BenchmarkRunner",
    "BenchmarkSummary",
    "QuestionBenchmarkResult",
    "ReportGenerator",
]
