"""
Retrieval Metrics Module.

Implements exact, deterministic evaluation metrics conforming to Section 5.2 of the specification:
- Hit Rate@K: (1/N) * sum(I[R_k ∩ G != ∅])
- Recall@K: (1/N) * sum(|R_k ∩ G| / |G|)
- Mean Reciprocal Rank (MRR): (1/N) * sum(1 / r)
- Precision@K: (1/N) * sum(|R_k ∩ G| / K)

Explicitly handles duplicates, missing labels, and fewer-than-K results.
"""

from __future__ import annotations

from typing import List, Set, Optional, Dict, Any, Sequence
from dataclasses import dataclass, field


@dataclass
class QueryRetrievalResult:
    """Individual query evaluation result."""
    question_id: str
    k: int
    retrieved_chunk_ids: List[str]
    ground_truth_chunk_ids: List[str]
    hit: bool
    recall: float
    reciprocal_rank: float
    precision: float
    first_relevant_rank: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "question_id": self.question_id,
            "k": self.k,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "ground_truth_chunk_ids": self.ground_truth_chunk_ids,
            "hit": int(self.hit),
            "recall": round(self.recall, 4),
            "reciprocal_rank": round(self.reciprocal_rank, 4),
            "precision": round(self.precision, 4),
            "first_relevant_rank": self.first_relevant_rank if self.first_relevant_rank is not None else "N/A"
        }


@dataclass
class RetrievalMetrics:
    """Aggregate benchmark retrieval metrics across N questions."""
    k: int
    num_questions: int
    hit_rate: float
    mean_recall: float
    mrr: float
    mean_precision: float
    num_hits: int
    per_query_results: List[QueryRetrievalResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert aggregate metrics to dictionary representation."""
        return {
            "k": self.k,
            "num_questions": self.num_questions,
            "hit_rate_at_k": round(self.hit_rate, 4),
            "recall_at_k": round(self.mean_recall, 4),
            "mrr": round(self.mrr, 4),
            "precision_at_k": round(self.mean_precision, 4),
            "num_hits": self.num_hits,
            "per_query_results": [r.to_dict() for r in self.per_query_results]
        }


def _deduplicate_preserve_order(sequence: Sequence[str]) -> List[str]:
    """
    Deduplicate retrieved chunk IDs while strictly preserving rank order.
    """
    seen: Set[str] = set()
    deduped: List[str] = []
    for item in sequence:
        cleaned = str(item).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            deduped.append(cleaned)
    return deduped


def evaluate_single_query(
    question_id: str,
    retrieved_chunk_ids: Sequence[str],
    ground_truth_chunk_ids: Sequence[str],
    k: int,
    strict_k_denominator: bool = True
) -> QueryRetrievalResult:
    """
    Compute deterministic retrieval metrics for a single question.

    Args:
        question_id: Identifier of the benchmark question.
        retrieved_chunk_ids: List of retrieved chunk IDs in ranked order.
        ground_truth_chunk_ids: Set/List of annotated relevant chunk IDs.
        k: Cut-off rank.
        strict_k_denominator: If True, Precision uses exact K in denominator as per formula.

    Returns:
        QueryRetrievalResult with hit, recall, reciprocal_rank, and precision.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")

    # Deduplicate retrieved IDs preserving rank order
    deduped_retrieved = _deduplicate_preserve_order(retrieved_chunk_ids)
    top_k_retrieved = deduped_retrieved[:k]

    # Clean ground truth set
    ground_truth_set = set(str(cid).strip() for cid in ground_truth_chunk_ids if str(cid).strip())

    # Case: Missing labels (empty ground truth)
    if not ground_truth_set:
        return QueryRetrievalResult(
            question_id=question_id,
            k=k,
            retrieved_chunk_ids=top_k_retrieved,
            ground_truth_chunk_ids=[],
            hit=False,
            recall=0.0,
            reciprocal_rank=0.0,
            precision=0.0,
            first_relevant_rank=None
        )

    # 1. Calculate Hit@K: I[R_k ∩ G != ∅]
    relevant_in_top_k = [cid for cid in top_k_retrieved if cid in ground_truth_set]
    is_hit = len(relevant_in_top_k) > 0

    # 2. Calculate Recall@K: |R_k ∩ G| / |G|
    recall = len(relevant_in_top_k) / len(ground_truth_set)

    # 3. Calculate Reciprocal Rank (1/r)
    first_rank: Optional[int] = None
    reciprocal_rank = 0.0
    for rank_1indexed, cid in enumerate(top_k_retrieved, start=1):
        if cid in ground_truth_set:
            first_rank = rank_1indexed
            reciprocal_rank = 1.0 / rank_1indexed
            break

    # 4. Calculate Precision@K: |R_k ∩ G| / K
    denominator = k if strict_k_denominator else max(len(top_k_retrieved), 1)
    precision = len(relevant_in_top_k) / denominator

    return QueryRetrievalResult(
        question_id=question_id,
        k=k,
        retrieved_chunk_ids=top_k_retrieved,
        ground_truth_chunk_ids=list(ground_truth_set),
        hit=is_hit,
        recall=recall,
        reciprocal_rank=reciprocal_rank,
        precision=precision,
        first_relevant_rank=first_rank
    )


def calculate_hit_rate_at_k(
    retrieved_lists: List[Sequence[str]],
    ground_truth_lists: List[Sequence[str]],
    k: int
) -> float:
    """
    HitRate@K = (1/N) * sum(I[R_k ∩ G != ∅])
    """
    if not retrieved_lists or len(retrieved_lists) != len(ground_truth_lists):
        return 0.0

    hits = 0
    n = len(retrieved_lists)
    for r, g in zip(retrieved_lists, ground_truth_lists):
        res = evaluate_single_query("tmp", r, g, k)
        if res.hit:
            hits += 1
    return hits / n


def calculate_recall_at_k(
    retrieved_lists: List[Sequence[str]],
    ground_truth_lists: List[Sequence[str]],
    k: int
) -> float:
    """
    Recall@K = (1/N) * sum(|R_k ∩ G| / |G|)
    """
    if not retrieved_lists or len(retrieved_lists) != len(ground_truth_lists):
        return 0.0

    total_recall = 0.0
    n = len(retrieved_lists)
    for r, g in zip(retrieved_lists, ground_truth_lists):
        res = evaluate_single_query("tmp", r, g, k)
        total_recall += res.recall
    return total_recall / n


def calculate_mrr(
    retrieved_lists: List[Sequence[str]],
    ground_truth_lists: List[Sequence[str]],
    k: int
) -> float:
    """
    MRR = (1/N) * sum(1 / r)
    """
    if not retrieved_lists or len(retrieved_lists) != len(ground_truth_lists):
        return 0.0

    total_rr = 0.0
    n = len(retrieved_lists)
    for r, g in zip(retrieved_lists, ground_truth_lists):
        res = evaluate_single_query("tmp", r, g, k)
        total_rr += res.reciprocal_rank
    return total_rr / n


def calculate_precision_at_k(
    retrieved_lists: List[Sequence[str]],
    ground_truth_lists: List[Sequence[str]],
    k: int,
    strict_k_denominator: bool = True
) -> float:
    """
    Precision@K = (1/N) * sum(|R_k ∩ G| / K)
    """
    if not retrieved_lists or len(retrieved_lists) != len(ground_truth_lists):
        return 0.0

    total_prec = 0.0
    n = len(retrieved_lists)
    for r, g in zip(retrieved_lists, ground_truth_lists):
        res = evaluate_single_query("tmp", r, g, k, strict_k_denominator=strict_k_denominator)
        total_prec += res.precision
    return total_prec / n


def evaluate_retrieval_batch(
    query_ids: List[str],
    retrieved_lists: List[Sequence[str]],
    ground_truth_lists: List[Sequence[str]],
    k: int = 5,
    strict_k_denominator: bool = True
) -> RetrievalMetrics:
    """
    Execute full deterministic retrieval metric evaluation over a batch of queries.

    Args:
        query_ids: List of question IDs.
        retrieved_lists: List of retrieved chunk IDs per question.
        ground_truth_lists: List of annotated ground-truth chunk IDs per question.
        k: Cut-off rank (default 5).
        strict_k_denominator: Flag for strict K precision division.

    Returns:
        RetrievalMetrics object containing both aggregate and per-query results.
    """
    n = len(query_ids)
    if n == 0:
        return RetrievalMetrics(
            k=k,
            num_questions=0,
            hit_rate=0.0,
            mean_recall=0.0,
            mrr=0.0,
            mean_precision=0.0,
            num_hits=0,
            per_query_results=[]
        )

    per_query_results: List[QueryRetrievalResult] = []
    total_hit = 0
    total_recall = 0.0
    total_rr = 0.0
    total_precision = 0.0

    for qid, ret, gt in zip(query_ids, retrieved_lists, ground_truth_lists):
        result = evaluate_single_query(
            question_id=qid,
            retrieved_chunk_ids=ret,
            ground_truth_chunk_ids=gt,
            k=k,
            strict_k_denominator=strict_k_denominator
        )
        per_query_results.append(result)
        if result.hit:
            total_hit += 1
        total_recall += result.recall
        total_rr += result.reciprocal_rank
        total_precision += result.precision

    return RetrievalMetrics(
        k=k,
        num_questions=n,
        hit_rate=total_hit / n,
        mean_recall=total_recall / n,
        mrr=total_rr / n,
        mean_precision=total_precision / n,
        num_hits=total_hit,
        per_query_results=per_query_results
    )
