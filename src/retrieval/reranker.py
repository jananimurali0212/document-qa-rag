"""
Reranker Module.

Reorders top-K candidate chunks using cross-encoder models or lexical-semantic
fusion to improve precision and contextual relevance before generation.
"""

from __future__ import annotations

import re
import math
from typing import List, Optional
from src.retrieval.vector_store import SearchResult


class Reranker:
    """
    Reranks candidate chunks using either a CrossEncoder neural model
    or a high-speed lexical-semantic score booster.
    """

    def __init__(
        self,
        model_name: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_neural: bool = False
    ):
        self.model_name = model_name
        self.use_neural = use_neural
        self._cross_encoder = None

        if self.use_neural and self.model_name:
            self._init_neural_model()

    def _init_neural_model(self) -> None:
        """Attempt to initialize sentence_transformers CrossEncoder."""
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(self.model_name)
        except Exception:
            self._cross_encoder = None
            self.use_neural = False

    STOP_WORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
        "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for",
        "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
        "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
        "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
        "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
        "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
        "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
        "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
        "why", "will", "with", "would", "you", "your", "yours", "yourself", "yourselves"
    }

    def _lexical_overlap_score(self, query: str, text: str) -> float:
        """
        Calculate token-level lexical overlap and content keyword density.
        """
        raw_q_tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", query.lower())
        content_q = [w for w in raw_q_tokens if w not in self.STOP_WORDS and len(w) > 1]
        active_q = set(content_q if content_q else raw_q_tokens)
        if not active_q:
            return 0.0

        t_text = text.lower()
        t_tokens = set(re.findall(r"\b[a-zA-Z0-9_]+\b", t_text))
        if not t_tokens:
            return 0.0

        # Exact and stem matches
        exact_matches = active_q.intersection(t_tokens)
        overlap_ratio = len(exact_matches) / len(active_q)

        # Partial/stem matches (e.g. "loop" in "loops")
        partial_matches = 0
        for q_word in active_q:
            if q_word not in exact_matches and any(q_word in t_word or t_word in q_word for t_word in t_tokens if len(t_word) > 3):
                partial_matches += 1
        partial_ratio = partial_matches / len(active_q)

        # Bigram phrase presence
        phrase_bonus = 0.0
        if len(content_q) >= 2:
            for i in range(len(content_q) - 1):
                phrase = f"{content_q[i]} {content_q[i+1]}"
                if phrase in t_text:
                    phrase_bonus += 0.25

        return min(1.0, (0.75 * overlap_ratio) + (0.15 * partial_ratio) + phrase_bonus)

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Re-score and re-rank candidate chunks for the given query.

        Args:
            query: User's question.
            candidates: Initial candidates from vector similarity search.
            top_k: Number of candidates to retain after reranking.

        Returns:
            List of SearchResult reordered by fused/reranked score descending.
        """
        if not candidates:
            return []

        # Neural CrossEncoder branch
        if self.use_neural and self._cross_encoder is not None:
            try:
                pairs = [[query, c.text] for c in candidates]
                scores = self._cross_encoder.predict(pairs)
                reranked: List[SearchResult] = []
                for candidate, score in zip(candidates, scores):
                    # Sigmoid or min-max normalization
                    norm_score = 1.0 / (1.0 + math.exp(-float(score)))
                    candidate.score = norm_score
                    reranked.append(candidate)
                reranked.sort(key=lambda x: x.score, reverse=True)
                return reranked[:top_k]
            except Exception:
                pass  # Fall back to lexical-semantic fusion

        # Lexical-semantic fusion branch
        reranked_candidates: List[SearchResult] = []
        for candidate in candidates:
            lex_score = self._lexical_overlap_score(query, candidate.text)
            # Fused score: 65% dense vector score + 35% lexical keyword score
            fused_score = (0.65 * candidate.score) + (0.35 * lex_score)
            candidate.score = round(fused_score, 4)
            reranked_candidates.append(candidate)

        reranked_candidates.sort(key=lambda x: x.score, reverse=True)
        return reranked_candidates[:top_k]
