"""
Report Generator Module.

Exports benchmark and evaluation results to CSV, JSON, and comprehensive Markdown reports.
Enables cross-configuration comparison (e.g. varying K, chunk sizes, or models).
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from src.evaluation.benchmark_runner import BenchmarkSummary, QuestionBenchmarkResult


class ReportGenerator:
    """
    Handles serialization, tabular formatting, and report artifact generation.
    """

    @staticmethod
    def to_dataframe(summary: BenchmarkSummary) -> pd.DataFrame:
        """
        Convert benchmark results to pandas DataFrame matching Section 5.5 format:
        Question | Hit@K | Recall@K | MRR | Faithfulness | Relevance | Correctness | Latency_ms
        """
        rows = [r.to_dict() for r in summary.per_question_results]
        return pd.DataFrame(rows)

    @staticmethod
    def export_csv(summary: BenchmarkSummary, output_path: Optional[str] = None) -> str:
        """
        Export per-question evaluation results as CSV text or save to file.
        """
        df = ReportGenerator.to_dataframe(summary)
        csv_str = df.to_csv(index=False)
        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(csv_str, encoding="utf-8")
        return csv_str

    @staticmethod
    def export_json(summary: BenchmarkSummary, output_path: Optional[str] = None) -> str:
        """
        Export complete benchmark summary and detailed questions as JSON string or save to file.
        """
        json_str = json.dumps(summary.to_dict(), indent=2, ensure_ascii=False)
        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json_str, encoding="utf-8")
        return json_str

    @staticmethod
    def generate_markdown_report(summary: BenchmarkSummary) -> str:
        """
        Generate a publication-ready Markdown report of evaluation findings.
        """
        faith_str = f"{summary.mean_faithfulness:.4f}" if summary.mean_faithfulness is not None else "N/A"
        rel_str = f"{summary.mean_relevance:.4f}" if summary.mean_relevance is not None else "N/A"
        corr_str = f"{summary.mean_correctness:.4f}" if summary.mean_correctness is not None else "N/A"

        md = f"""# Document QA RAG — Evaluation Benchmark Report

- **Experiment ID**: `{summary.experiment_id}`
- **Timestamp**: {summary.timestamp}
- **Benchmark Sample Size**: {summary.num_questions} questions
- **Evaluation Cutoff**: Top-K = {summary.k}

---

## 1. Executive Summary & Aggregate Metrics

| Metric | Measured Score | Target / Reference | Description |
| :--- | :--- | :--- | :--- |
| **Hit Rate@{summary.k}** | **{summary.hit_rate_at_k:.4f}** ({summary.hit_rate_at_k*100:.1f}%) | ≥ 80% | Fraction of questions where at least 1 relevant chunk was retrieved |
| **Recall@{summary.k}** | **{summary.recall_at_k:.4f}** | ≥ 70% | Average fraction of annotated ground-truth chunks retrieved |
| **Mean Reciprocal Rank (MRR)** | **{summary.mrr:.4f}** | ≥ 0.70 | Measures how early the first relevant result appears in ranking |
| **Precision@{summary.k}** | **{summary.precision_at_k:.4f}** | — | Fraction of returned candidates that are relevant |
| **Faithfulness** | **{faith_str}** | ≥ 0.85 | Degree to which generated answers are strictly grounded in context |
| **Answer Relevance** | **{rel_str}** | ≥ 0.80 | Alignment of generated answer with the question intent |
| **Answer Correctness** | **{corr_str}** | ≥ 0.75 | Agreement with golden reference answers |
| **Average Latency** | **{summary.avg_latency_ms:.1f} ms** | < 1500 ms | Average end-to-end question answering latency |

---

## 2. Per-Question Results Table

| Question ID | Query | Hit@{summary.k} | Recall@{summary.k} | MRR | Precision@{summary.k} | Faithfulness | Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
        for r in summary.per_question_results:
            q_snippet = (r.question[:35] + "...") if len(r.question) > 35 else r.question
            faith = f"{r.faithfulness:.2f}" if r.faithfulness is not None else "N/A"
            md += f"| `{r.question_id}` | {q_snippet} | {r.hit_at_k} | {r.recall_at_k:.2f} | {r.mrr:.2f} | {r.precision_at_k:.2f} | {faith} | {r.latency_ms:.1f} |\n"

        md += "\n---\n*Generated automatically by Document QA RAG Benchmark Harness.*\n"
        return md

    @staticmethod
    def compare_experiments(summaries: List[BenchmarkSummary]) -> pd.DataFrame:
        """
        Produce a comparison table across multiple benchmark configurations.
        """
        rows = []
        for s in summaries:
            rows.append({
                "Experiment": s.experiment_id,
                "K": s.k,
                "Questions": s.num_questions,
                "Hit_Rate@K": round(s.hit_rate_at_k, 4),
                "Recall@K": round(s.recall_at_k, 4),
                "MRR": round(s.mrr, 4),
                "Precision@K": round(s.precision_at_k, 4),
                "Faithfulness": round(s.mean_faithfulness, 4) if s.mean_faithfulness is not None else "N/A",
                "Relevance": round(s.mean_relevance, 4) if s.mean_relevance is not None else "N/A",
                "Avg_Latency_ms": round(s.avg_latency_ms, 2)
            })
        return pd.DataFrame(rows)
