"""
Metadata Store Module.

Persists document records, chunk indices, and benchmark experiment results
using an embedded SQLite database for reproducibility and audit trails.
"""

from __future__ import annotations

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.ingestion.chunker import ChunkRecord
from src.ingestion.file_validator import IngestionStatus


class MetadataStore:
    """
    SQLite-backed repository for document metadata, chunk records,
    and historical evaluation benchmarks.
    """

    def __init__(self, db_path: str = "rag_metadata.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create necessary tables if they do not exist."""
        parent_dir = Path(self.db_path).parent
        if str(parent_dir) not in ("", "."):
            parent_dir.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Documents Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    document_name TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    num_pages INTEGER NOT NULL,
                    num_chunks INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            # Chunks Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    document_name TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents (document_id) ON DELETE CASCADE
                )
            """)

            # Experiments Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    k INTEGER NOT NULL,
                    num_questions INTEGER NOT NULL,
                    hit_rate_at_k REAL NOT NULL,
                    recall_at_k REAL NOT NULL,
                    mrr REAL NOT NULL,
                    precision_at_k REAL NOT NULL,
                    faithfulness REAL,
                    avg_latency_ms REAL NOT NULL,
                    summary_json TEXT NOT NULL
                )
            """)
            conn.commit()

    def record_document(
        self,
        document_id: str,
        document_name: str,
        file_type: str,
        file_size_bytes: int,
        num_pages: int,
        num_chunks: int,
        status: IngestionStatus = IngestionStatus.INDEXED
    ) -> None:
        """Upsert a document record."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO documents (document_id, document_name, file_type, file_size_bytes, num_pages, num_chunks, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    status = excluded.status,
                    num_chunks = excluded.num_chunks,
                    num_pages = excluded.num_pages
            """, (document_id, document_name, file_type, file_size_bytes, num_pages, num_chunks, status.value, now))
            conn.commit()

    def record_chunks(self, chunks: List[ChunkRecord]) -> None:
        """Batch insert chunk records."""
        if not chunks:
            return
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO chunks (chunk_id, document_id, document_name, page_number, chunk_index, text)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                (c.chunk_id, c.document_id, c.document_name, c.page_number, c.chunk_index, c.text)
                for c in chunks
            ])
            conn.commit()

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retrieve all ingested document records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_all_chunks(self) -> List[ChunkRecord]:
        """Retrieve all chunks across all documents as ChunkRecord objects."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chunks ORDER BY document_id, chunk_index ASC")
            return [
                ChunkRecord(
                    chunk_id=r["chunk_id"],
                    document_id=r["document_id"],
                    document_name=r["document_name"],
                    page_number=r["page_number"],
                    chunk_index=r["chunk_index"],
                    text=r["text"]
                )
                for r in cursor.fetchall()
            ]

    def get_chunks_for_document(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks belonging to a document."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC", (document_id,))
            return [dict(row) for row in cursor.fetchall()]

    def delete_document(self, document_id: str) -> bool:
        """Delete document and all associated chunks."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            cursor.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            conn.commit()
            return cursor.rowcount > 0

    def record_experiment(
        self,
        experiment_id: str,
        k: int,
        num_questions: int,
        hit_rate_at_k: float,
        recall_at_k: float,
        mrr: float,
        precision_at_k: float,
        faithfulness: Optional[float],
        avg_latency_ms: float,
        summary_dict: Dict[str, Any]
    ) -> None:
        """Store benchmark experiment run results."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO experiments (
                    experiment_id, timestamp, k, num_questions, hit_rate_at_k,
                    recall_at_k, mrr, precision_at_k, faithfulness, avg_latency_ms, summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id, now, k, num_questions, hit_rate_at_k,
                recall_at_k, mrr, precision_at_k, faithfulness, avg_latency_ms,
                json.dumps(summary_dict)
            ))
            conn.commit()

    def get_experiments(self) -> List[Dict[str, Any]]:
        """Retrieve past benchmark experiment runs."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiments ORDER BY timestamp DESC")
            return [dict(row) for row in cursor.fetchall()]

    def clear_all(self) -> None:
        """Clear all documents, chunks, and experiment history."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks")
            cursor.execute("DELETE FROM documents")
            cursor.execute("DELETE FROM experiments")
            conn.commit()
