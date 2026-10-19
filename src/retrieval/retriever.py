"""
Retriever Module.

Orchestrates query embedding, vector similarity search, and optional reranking
to return the most relevant document chunks for answering questions.
"""

from __future__ import annotations

import time
from typing import List, Optional
from src.embeddings.embedding_service import EmbeddingService
from src.retrieval.vector_store import VectorStore, SearchResult
from src.retrieval.reranker import Reranker


class Retriever:
    """
    RAG Retriever combining dense query embedding, vector similarity lookup,
    and optional cross-encoder / fusion reranking.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        reranker: Optional[Reranker] = None
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_multiplier: int = 2,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Retrieve the top-K most relevant chunks for a given query.

        Args:
            query: User's natural language question.
            top_k: Number of final chunks to return.
            candidate_multiplier: Factor of top_k to fetch before reranking.
            score_threshold: Optional minimum similarity score cutoff.

        Returns:
            List of SearchResult objects ordered by relevance.
        """
        if not query or not query.strip():
            return []

        # 1. Embed query
        query_vec = self.embedding_service.embed_query(query)

        # 2. Determine initial candidate pool size
        initial_k = top_k * candidate_multiplier if self.reranker is not None else top_k

        # 3. Vector search
        candidates = self.vector_store.search(query_vec, top_k=initial_k)
        if not candidates:
            return []

        # 4. Optional Reranking
        if self.reranker is not None:
            candidates = self.reranker.rerank(query=query, candidates=candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        # 5. Optional threshold filtering
        if score_threshold is not None:
            candidates = [c for c in candidates if c.score >= score_threshold]

        return candidates
