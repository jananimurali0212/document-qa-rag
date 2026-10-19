"""
Unit Tests for Retrieval Metrics.

Validates exact mathematical calculations against hand-crafted examples for:
- Hit Rate@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Precision@K
And tests edge cases: duplicates, empty retrieval, missing labels, and fewer-than-K results.
"""

import pytest
from src.evaluation.retrieval_metrics import (
    evaluate_single_query,
    evaluate_retrieval_batch,
    calculate_hit_rate_at_k,
    calculate_recall_at_k,
    calculate_mrr,
    calculate_precision_at_k,
)


class TestRetrievalMetricsHandCalculated:
    """Verifies metrics against hand-calculated ground truth cases."""

    def test_single_query_perfect_retrieval(self):
        # Ground truth has 2 chunks; both retrieved at ranks 1 and 2
        retrieved = ["doc01_chunk_002", "doc01_chunk_003", "irrelevant_1", "irrelevant_2", "irrelevant_3"]
        ground_truth = ["doc01_chunk_002", "doc01_chunk_003"]
        k = 5

        res = evaluate_single_query("Q001", retrieved, ground_truth, k=k)

        assert res.hit is True
        assert res.recall == 1.0  # 2 / 2
        assert res.reciprocal_rank == 1.0  # 1 / 1
        assert res.precision == 2.0 / 5.0  # 2 / 5 = 0.4
        assert res.first_relevant_rank == 1

    def test_single_query_partial_retrieval_rank_two(self):
        # Ground truth has 2 chunks; only 1 retrieved at rank 2
        retrieved = ["irrelevant_1", "doc01_chunk_002", "irrelevant_2", "irrelevant_3", "irrelevant_4"]
        ground_truth = ["doc01_chunk_002", "doc01_chunk_003"]
        k = 5

        res = evaluate_single_query("Q002", retrieved, ground_truth, k=k)

        assert res.hit is True
        assert res.recall == 0.5  # 1 / 2
        assert res.reciprocal_rank == 0.5  # 1 / 2
        assert res.precision == 0.2  # 1 / 5
        assert res.first_relevant_rank == 2

    def test_single_query_complete_miss(self):
        retrieved = ["irrelevant_1", "irrelevant_2", "irrelevant_3"]
        ground_truth = ["doc01_chunk_009"]
        k = 5

        res = evaluate_single_query("Q003", retrieved, ground_truth, k=k)

        assert res.hit is False
        assert res.recall == 0.0
        assert res.reciprocal_rank == 0.0
        assert res.precision == 0.0
        assert res.first_relevant_rank is None

    def test_batch_aggregate_metrics(self):
        # 3 questions corresponding to Example in spec Section 5.5
        q_ids = ["Q001", "Q002", "Q003"]
        retrieved_batch = [
            ["doc01_chunk_002", "doc01_chunk_003", "other1", "other2", "other3"],
            ["other1", "doc01_chunk_002", "other2", "other3", "other4"],
            ["other1", "other2", "other3", "other4", "other5"]
        ]
        gt_batch = [
            ["doc01_chunk_002", "doc01_chunk_003"],
            ["doc01_chunk_002", "doc01_chunk_003"],
            ["doc01_chunk_009"]
        ]
        k = 5

        summary = evaluate_retrieval_batch(q_ids, retrieved_batch, gt_batch, k=k)

        # Expected:
        # Hits: 2 out of 3 -> 2/3 ≈ 0.6667
        # Recall: (1.0 + 0.5 + 0.0) / 3 = 0.5000
        # MRR: (1.0 + 0.5 + 0.0) / 3 = 0.5000
        # Precision: (0.4 + 0.2 + 0.0) / 3 = 0.2000
        assert summary.num_questions == 3
        assert summary.num_hits == 2
        assert pytest.approx(summary.hit_rate, 0.0001) == 2.0 / 3.0
        assert pytest.approx(summary.mean_recall, 0.0001) == 0.5000
        assert pytest.approx(summary.mrr, 0.0001) == 0.5000
        assert pytest.approx(summary.mean_precision, 0.0001) == 0.2000


class TestRetrievalMetricsEdgeCases:
    """Verifies edge cases required by the specification."""

    def test_duplicate_retrieved_results_deduplicated_preserving_order(self):
        # Duplicate relevant ID should not falsely inflate recall or precision
        retrieved = ["doc01_chunk_002", "doc01_chunk_002", "irrelevant_1"]
        ground_truth = ["doc01_chunk_002"]
        k = 3

        res = evaluate_single_query("Q_dup", retrieved, ground_truth, k=k)

        # Only 1 unique match
        assert res.hit is True
        assert res.recall == 1.0  # 1 / 1
        assert res.precision == 1.0 / 3.0  # 1 / 3
        assert res.reciprocal_rank == 1.0

    def test_empty_retrieval_list(self):
        res = evaluate_single_query("Q_empty", [], ["doc01_chunk_001"], k=5)
        assert res.hit is False
        assert res.recall == 0.0
        assert res.reciprocal_rank == 0.0
        assert res.precision == 0.0
        assert res.first_relevant_rank is None

    def test_missing_ground_truth_labels(self):
        # When dataset has no annotated ground truth for a question
        res = evaluate_single_query("Q_nolabel", ["doc01_chunk_001"], [], k=5)
        assert res.hit is False
        assert res.recall == 0.0
        assert res.reciprocal_rank == 0.0
        assert res.precision == 0.0

    def test_fewer_than_k_results_returned(self):
        # Database only has 2 chunks, but k=5 requested
        retrieved = ["doc01_chunk_001", "irrelevant"]
        ground_truth = ["doc01_chunk_001"]
        k = 5

        res = evaluate_single_query("Q_short", retrieved, ground_truth, k=k, strict_k_denominator=True)
        assert res.hit is True
        assert res.recall == 1.0
        assert res.precision == 1.0 / 5.0  # strict denominator K=5
