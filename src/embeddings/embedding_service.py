"""
Embedding Service Module.

Encodes chunks and queries into dense vector representations using Sentence Transformers
(e.g., all-MiniLM-L6-v2), with automatic fallbacks for offline testing and graceful degradation.
"""

from __future__ import annotations

import os
import re
import math
import hashlib
from typing import List, Union, Optional
import numpy as np
from src.ingestion.chunker import ChunkRecord


class EmbeddingService:
    """
    Modular embedding service supporting Sentence Transformers and deterministic fallbacks.
    Vectors are L2-normalized to ensure inner product corresponds to cosine similarity.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_DIMENSION = 384

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        dimension: int = DEFAULT_DIMENSION,
        use_fallback_if_unavailable: bool = True
    ):
        """
        Initialize the embedding service.

        Args:
            model_name: HuggingFace sentence-transformers model name.
            dimension: Vector embedding dimension (default 384 for all-MiniLM-L6-v2).
            use_fallback_if_unavailable: If True, falls back to a deterministic hashing
                                         embedding if sentence-transformers is missing or offline.
        """
        self.model_name = model_name
        self.dimension = dimension
        self.use_fallback_if_unavailable = use_fallback_if_unavailable
        self._model = None
        self._is_fallback = False

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Attempt to load SentenceTransformer model; fallback gracefully if not found."""
        try:
            from sentence_transformers import SentenceTransformer
            # Load model
            self._model = SentenceTransformer(self.model_name)
            self._is_fallback = False
            # Verify actual dimension
            test_emb = self._model.encode(["test"], show_progress_bar=False)
            self.dimension = test_emb.shape[1]
        except Exception:
            if self.use_fallback_if_unavailable:
                self._is_fallback = True
                self._model = None
            else:
                raise

    @property
    def is_fallback(self) -> bool:
        """Indicate whether the deterministic fallback embedder is currently active."""
        return self._is_fallback

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

    def _deterministic_hash_embed(self, text: str) -> np.ndarray:
        """
        Produce a deterministic, normalized 384-dimensional dense vector from text
        using hashed word and character n-grams with content-word weighting.
        """
        vec = np.zeros(self.dimension, dtype=np.float32)
        if not text:
            vec[0] = 1.0
            return vec

        raw_tokens = [w.strip() for w in re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower()) if len(w.strip()) > 1]
        if not raw_tokens:
            vec[0] = 1.0
            return vec

        for idx, word in enumerate(raw_tokens):
            token_bytes = word.encode("utf-8")
            h1 = int(hashlib.md5(token_bytes).hexdigest(), 16) % self.dimension
            h2 = int(hashlib.sha1(token_bytes).hexdigest(), 16) % self.dimension

            # Substantially boost content words over stop words
            is_content = word not in self.STOP_WORDS
            base_weight = 3.5 if is_content else 0.25

            weight = base_weight / (1.0 + 0.1 * math.log(1.0 + idx))
            vec[h1] += weight
            vec[h2] -= weight * 0.5

            # Bigram hashing for adjacent content words
            if idx > 0:
                prev = raw_tokens[idx - 1]
                if is_content or prev not in self.STOP_WORDS:
                    bigram = f"{prev}_{word}".encode("utf-8")
                    hb = int(hashlib.sha256(bigram).hexdigest(), 16) % self.dimension
                    vec[hb] += 2.0

        # Subword trigram hashing for partial matches
        clean_text = text.lower()
        for i in range(min(len(clean_text) - 2, 400)):
            trigram = clean_text[i:i + 3].encode("utf-8")
            idx = int(hashlib.blake2s(trigram, digest_size=2).hexdigest(), 16) % self.dimension
            vec[idx] += 0.2

        # L2 Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec[0] = 1.0

        return vec

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of document strings into a 2D float32 numpy array (N, D).
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if self._model is not None and not self._is_fallback:
            try:
                embeddings = self._model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                return embeddings.astype(np.float32)
            except Exception:
                if not self.use_fallback_if_unavailable:
                    raise

        # Fallback deterministic embedding
        matrix = np.array([self._deterministic_hash_embed(t) for t in texts], dtype=np.float32)
        return matrix

    def embed_chunks(self, chunks: List[ChunkRecord]) -> np.ndarray:
        """
        Extract text from ChunkRecords and return dense embedding matrix.
        """
        texts = [c.text for c in chunks]
        return self.embed_documents(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single search query into a 1D float32 vector (D,).
        """
        if self._model is not None and not self._is_fallback:
            try:
                emb = self._model.encode(
                    [query],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                return emb[0].astype(np.float32)
            except Exception:
                if not self.use_fallback_if_unavailable:
                    raise

        return self._deterministic_hash_embed(query)
