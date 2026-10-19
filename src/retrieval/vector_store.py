"""
Vector Store Module.

Manages vector embeddings and chunk metadata. Supports FAISS IndexFlatIP (cosine similarity)
with high-speed pure NumPy fallback, persistence to disk, and document lifecycle deletion.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
from src.ingestion.chunker import ChunkRecord


@dataclass
class SearchResult:
    """Retrieved chunk candidate with relevance score and complete metadata."""
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    chunk_index: int
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert SearchResult to serializable dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "score": round(self.score, 4),
            "metadata": self.metadata
        }


class VectorStore:
    """
    Vector storage and approximate/exact nearest neighbor search.
    Supports FAISS if installed, with seamless NumPy cosine similarity fallback.
    """

    def __init__(self, dimension: int = 384, use_faiss: bool = True):
        self.dimension = dimension
        self.use_faiss = use_faiss
        self.faiss_available = False
        self._faiss_index = None

        # Metadata mapping: chunk_id -> ChunkRecord
        self.chunk_records: Dict[str, ChunkRecord] = {}
        # Ordered list of chunk_ids corresponding to row indices in self.embeddings
        self.ordered_chunk_ids: List[str] = []
        # Embedding matrix (N, dimension)
        self.embeddings: Optional[np.ndarray] = None

        if self.use_faiss:
            self._init_faiss()

    def _init_faiss(self) -> None:
        """Initialize FAISS IndexFlatIP for normalized cosine similarity search."""
        try:
            import faiss
            self._faiss_index = faiss.IndexFlatIP(self.dimension)
            self.faiss_available = True
        except ImportError:
            self.faiss_available = False
            self._faiss_index = None

    def add_chunks(self, chunks: List[ChunkRecord], embeddings: np.ndarray) -> None:
        """
        Add chunks and their dense embeddings to the index.

        Args:
            chunks: List of ChunkRecords matching the rows of embeddings.
            embeddings: Float32 numpy array of shape (len(chunks), dimension).
        """
        if len(chunks) == 0:
            return

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Chunks count ({len(chunks)}) does not match embeddings rows ({embeddings.shape[0]})"
            )

        # Normalize embeddings to unit length for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_embs = (embeddings / norms).astype(np.float32)

        for chunk in chunks:
            self.chunk_records[chunk.chunk_id] = chunk
            self.ordered_chunk_ids.append(chunk.chunk_id)

        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = normalized_embs
        else:
            self.embeddings = np.vstack([self.embeddings, normalized_embs])

        if self.faiss_available and self._faiss_index is not None:
            self._faiss_index.add(normalized_embs)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """
        Perform top-K similarity search against indexed chunks.

        Args:
            query_vector: 1D query embedding (dimension,).
            top_k: Number of nearest chunks to return.

        Returns:
            List of SearchResult objects sorted by score descending.
        """
        if self.embeddings is None or len(self.ordered_chunk_ids) == 0:
            return []

        # Ensure query is 2D and normalized
        q = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        top_k = min(top_k, len(self.ordered_chunk_ids))

        # FAISS search
        if self.faiss_available and self._faiss_index is not None:
            try:
                distances, indices = self._faiss_index.search(q, top_k)
                results: List[SearchResult] = []
                for score, idx in zip(distances[0], indices[0]):
                    if idx < 0 or idx >= len(self.ordered_chunk_ids):
                        continue
                    chunk_id = self.ordered_chunk_ids[idx]
                    chunk = self.chunk_records[chunk_id]
                    results.append(
                        SearchResult(
                            chunk_id=chunk.chunk_id,
                            document_id=chunk.document_id,
                            document_name=chunk.document_name,
                            page_number=chunk.page_number,
                            chunk_index=chunk.chunk_index,
                            text=chunk.text,
                            score=float(score),
                            metadata=chunk.metadata or {}
                        )
                    )
                return results
            except Exception:
                # Fallback to numpy if FAISS search errors
                pass

        # NumPy cosine similarity search
        scores = np.dot(self.embeddings, q[0])
        # Get top-k indices
        if len(scores) <= top_k:
            ranked_indices = np.argsort(-scores)
        else:
            top_partition = np.argpartition(-scores, top_k)[:top_k]
            ranked_indices = top_partition[np.argsort(-scores[top_partition])]

        results: List[SearchResult] = []
        for idx in ranked_indices:
            chunk_id = self.ordered_chunk_ids[idx]
            chunk = self.chunk_records[chunk_id]
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    score=float(scores[idx]),
                    metadata=chunk.metadata or {}
                )
            )

        return results

    def delete_document(self, document_id: str) -> int:
        """
        Delete all vectors and metadata belonging to document_id.
        Rebuilds index to ensure consistency.

        Returns:
            Number of chunks deleted.
        """
        ids_to_remove = [
            cid for cid, chunk in self.chunk_records.items()
            if chunk.document_id == document_id
        ]
        if not ids_to_remove:
            return 0

        # Filter chunk records
        for cid in ids_to_remove:
            self.chunk_records.pop(cid, None)

        # Retain indices not in ids_to_remove
        keep_indices = [
            i for i, cid in enumerate(self.ordered_chunk_ids)
            if cid not in ids_to_remove
        ]

        if keep_indices and self.embeddings is not None:
            self.ordered_chunk_ids = [self.ordered_chunk_ids[i] for i in keep_indices]
            self.embeddings = self.embeddings[keep_indices]
        else:
            self.ordered_chunk_ids = []
            self.embeddings = None

        # Re-initialize FAISS index
        if self.faiss_available:
            self._init_faiss()
            if self.embeddings is not None and len(self.embeddings) > 0:
                self._faiss_index.add(self.embeddings)

        return len(ids_to_remove)

    def clear(self) -> None:
        """Clear all indexed documents and embeddings."""
        self.chunk_records.clear()
        self.ordered_chunk_ids.clear()
        self.embeddings = None
        if self.faiss_available:
            self._init_faiss()

    def get_chunk_by_id(self, chunk_id: str) -> Optional[ChunkRecord]:
        """Lookup chunk by ID."""
        return self.chunk_records.get(chunk_id)

    def get_all_chunks(self) -> List[ChunkRecord]:
        """Return all indexed chunk records."""
        return list(self.chunk_records.values())

    def get_stats(self) -> Dict[str, Any]:
        """Summary of store contents and backend."""
        doc_ids = set(c.document_id for c in self.chunk_records.values())
        return {
            "total_chunks": len(self.chunk_records),
            "total_documents": len(doc_ids),
            "dimension": self.dimension,
            "backend": "FAISS IndexFlatIP" if (self.faiss_available and self._faiss_index is not None) else "NumPy Exact Cosine",
            "document_ids": list(doc_ids)
        }

    def save(self, directory_path: str) -> None:
        """Save vector store metadata and embeddings to directory."""
        dir_path = Path(directory_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        meta = {
            "dimension": self.dimension,
            "ordered_chunk_ids": self.ordered_chunk_ids,
            "chunks": {cid: c.to_dict() for cid, c in self.chunk_records.items()}
        }
        (dir_path / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if self.embeddings is not None:
            np.save(str(dir_path / "embeddings.npy"), self.embeddings)

    def load(self, directory_path: str) -> None:
        """Load vector store metadata and embeddings from directory."""
        dir_path = Path(directory_path)
        meta_file = dir_path / "metadata.json"
        emb_file = dir_path / "embeddings.npy"

        if not meta_file.exists():
            raise FileNotFoundError(f"Missing vector store metadata: {meta_file}")

        data = json.loads(meta_file.read_text(encoding="utf-8"))
        self.dimension = data.get("dimension", self.dimension)
        self.ordered_chunk_ids = data.get("ordered_chunk_ids", [])
        self.chunk_records = {
            cid: ChunkRecord.from_dict(cdata)
            for cid, cdata in data.get("chunks", {}).items()
        }

        if emb_file.exists():
            self.embeddings = np.load(str(emb_file))
        else:
            self.embeddings = None

        if self.faiss_available:
            self._init_faiss()
            if self.embeddings is not None and len(self.embeddings) > 0:
                self._faiss_index.add(self.embeddings)
