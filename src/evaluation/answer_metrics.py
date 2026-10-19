"""
Answer Metrics Module.

Evaluates generation quality according to Section 5.3 of the specification:
- Faithfulness: verifies if answer claims are grounded in retrieved context.
- Answer Relevance: verifies if the answer addresses the question.
- Answer Correctness: agreement with reference answers.
- Context Precision & Context Recall.

Includes RAGAS framework interface hook and deterministic fallback evaluation.
"""

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from src.retrieval.vector_store import SearchResult


@dataclass
class AnswerQualityScore:
    """Individual question answer quality evaluation."""
    question_id: str
    faithfulness: Optional[float]
    answer_relevance: Optional[float]
    answer_correctness: Optional[float]
    context_precision: Optional[float]
    context_recall: Optional[float]
    evaluator_mode: str  # "ragas", "gemini_judge", or "deterministic_nlp"

    def to_dict(self) -> Dict[str, Any]:
        """Convert scores to dictionary."""
        return {
            "question_id": self.question_id,
            "faithfulness": round(self.faithfulness, 4) if self.faithfulness is not None else "N/A",
            "answer_relevance": round(self.answer_relevance, 4) if self.answer_relevance is not None else "N/A",
            "answer_correctness": round(self.answer_correctness, 4) if self.answer_correctness is not None else "N/A",
            "context_precision": round(self.context_precision, 4) if self.context_precision is not None else "N/A",
            "context_recall": round(self.context_recall, 4) if self.context_recall is not None else "N/A",
            "evaluator_mode": self.evaluator_mode
        }


class AnswerMetricsEvaluator:
    """
    Evaluator for RAG answer quality.
    Supports RAGAS framework integration and zero-dependency deterministic NLP evaluation.
    """

    def __init__(self, mode: str = "auto", gemini_api_key: Optional[str] = None):
        """
        Initialize the evaluator.

        Args:
            mode: "ragas", "deterministic", or "auto" (attempts ragas, falls back to deterministic).
            gemini_api_key: Optional API key if using LLM judge.
        """
        self.mode = mode
        self.gemini_api_key = gemini_api_key
        self.ragas_available = False

        if self.mode in ("auto", "ragas"):
            self._check_ragas()

    def _check_ragas(self) -> None:
        """Check if RAGAS is installed and importable."""
        try:
            import ragas
            self.ragas_available = True
        except ImportError:
            self.ragas_available = False

    def _tokenize(self, text: str) -> List[str]:
        """Simple alphanumeric tokenizer."""
        return re.findall(r"\w+", text.lower())

    def _compute_token_f1(self, pred: str, ref: str) -> float:
        """Calculate token-level precision, recall, and F1 score."""
        p_toks = self._tokenize(pred)
        r_toks = self._tokenize(ref)
        if not p_toks or not r_toks:
            return 0.0

        common = 0
        r_counts: Dict[str, int] = {}
        for t in r_toks:
            r_counts[t] = r_counts.get(t, 0) + 1

        for t in p_toks:
            if r_counts.get(t, 0) > 0:
                common += 1
                r_counts[t] -= 1

        if common == 0:
            return 0.0

        p = common / len(p_toks)
        r = common / len(r_toks)
        return (2 * p * r) / (p + r)

    def evaluate_deterministic(
        self,
        question_id: str,
        question: str,
        answer: str,
        retrieved_contexts: List[str],
        reference_answer: Optional[str] = None,
        retrieval_hit: bool = True
    ) -> AnswerQualityScore:
        """
        Compute deterministic faithfulness, relevance, and correctness scores.
        Conforming to rule: "Use N/A when answer-quality evaluation is not applicable or was not run;
        do not infer faithfulness solely from retrieval failure."
        """
        # If retrieval completely missed context or answer abstained
        if not retrieval_hit or not retrieved_contexts or not answer or "insufficient information" in answer.lower():
            # Return N/A (None) for faithfulness as explicitly stated in Section 5.5
            return AnswerQualityScore(
                question_id=question_id,
                faithfulness=None,
                answer_relevance=0.0 if not answer else 0.5,
                answer_correctness=0.0 if not reference_answer else self._compute_token_f1(answer, reference_answer),
                context_precision=0.0,
                context_recall=0.0,
                evaluator_mode="deterministic_nlp"
            )

        # 1. Faithfulness: decompose answer into sentences and verify support in retrieved contexts
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 10]
        context_corpus = " ".join(retrieved_contexts).lower()

        if not sentences:
            faithfulness_score = 0.5
        else:
            supported_count = 0
            for sent in sentences:
                sent_words = [w for w in self._tokenize(sent) if len(w) > 3]
                if not sent_words:
                    supported_count += 1
                    continue
                # Check fraction of content words appearing in context
                in_context = sum(1 for w in sent_words if w in context_corpus)
                overlap_ratio = in_context / len(sent_words)
                if overlap_ratio >= 0.60:
                    supported_count += 1

            faithfulness_score = round(supported_count / len(sentences), 4)

        # 2. Answer relevance: overlap between question tokens and answer tokens
        q_tokens = set(self._tokenize(question))
        a_tokens = set(self._tokenize(answer))
        overlap = len(q_tokens.intersection(a_tokens))
        relevance_score = round(min(1.0, (overlap / max(1, len(q_tokens))) * 1.5), 4)

        # 3. Answer correctness (F1 with reference answer)
        correctness_score = None
        if reference_answer:
            correctness_score = round(self._compute_token_f1(answer, reference_answer), 4)

        # 4. Context precision and recall heuristics
        context_precision = round(min(1.0, relevance_score * 0.9), 4)
        context_recall = round(min(1.0, (faithfulness_score + relevance_score) / 2.0), 4)

        return AnswerQualityScore(
            question_id=question_id,
            faithfulness=faithfulness_score,
            answer_relevance=relevance_score,
            answer_correctness=correctness_score,
            context_precision=context_precision,
            context_recall=context_recall,
            evaluator_mode="deterministic_nlp"
        )

    def evaluate_single(
        self,
        question_id: str,
        question: str,
        answer: str,
        retrieved_contexts: List[str],
        reference_answer: Optional[str] = None,
        retrieval_hit: bool = True
    ) -> AnswerQualityScore:
        """
        Evaluate single answer using the best available engine (RAGAS or deterministic).
        """
        if self.ragas_available and self.mode in ("ragas", "auto"):
            try:
                # If ragas is configured, attempt execution
                from ragas import evaluate
                from ragas.metrics import faithfulness, answer_relevancy
                from datasets import Dataset

                data = {
                    "question": [question],
                    "answer": [answer],
                    "contexts": [retrieved_contexts],
                    "ground_truth": [reference_answer or ""]
                }
                dataset = Dataset.from_dict(data)
                ragas_res = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
                return AnswerQualityScore(
                    question_id=question_id,
                    faithfulness=float(ragas_res["faithfulness"]),
                    answer_relevance=float(ragas_res["answer_relevancy"]),
                    answer_correctness=self._compute_token_f1(answer, reference_answer) if reference_answer else None,
                    context_precision=None,
                    context_recall=None,
                    evaluator_mode="ragas"
                )
            except Exception:
                # Fall back gracefully to deterministic evaluation
                pass

        return self.evaluate_deterministic(
            question_id=question_id,
            question=question,
            answer=answer,
            retrieved_contexts=retrieved_contexts,
            reference_answer=reference_answer,
            retrieval_hit=retrieval_hit
        )
