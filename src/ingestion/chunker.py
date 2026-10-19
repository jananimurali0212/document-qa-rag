"""
Chunking Engine Module.

Splits parsed documents into passages with configurable chunk size and overlap,
preserving page references and attaching metadata strictly conforming to the project schema.
"""

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from src.ingestion.document_parser import ParsedDocument, PageContent


@dataclass
class ChunkRecord:
    """
    Schema conforming strictly to specification:
    {
      "chunk_id": "doc01_chunk_004",
      "document_id": "doc01",
      "document_name": "guide.pdf",
      "page_number": 5,
      "text": "The system retrieves relevant passages...",
      "chunk_index": 4
    }
    """
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    text: str
    chunk_index: int
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk record to dictionary representation."""
        data = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "text": self.text,
            "chunk_index": self.chunk_index,
        }
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChunkRecord:
        """Construct ChunkRecord from dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            document_name=data["document_name"],
            page_number=data["page_number"],
            text=data["text"],
            chunk_index=data["chunk_index"],
            metadata=data.get("metadata")
        )


class Chunker:
    """
    Splits text passages into overlapping windows while respecting word boundaries
    and attaching required document/page citations.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        min_chunk_length: int = 40
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target character length for each chunk (default: 500).
            chunk_overlap: Number of characters to overlap between successive chunks (default: 100).
            min_chunk_length: Minimum characters for a chunk to avoid tiny trailing fragments.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

    def _split_text_with_overlap(self, text: str) -> List[str]:
        """
        Split a continuous string into chunks of chunk_size with chunk_overlap,
        snapping to whitespace or punctuation boundaries when possible.
        """
        if not text or len(text.strip()) == 0:
            return []

        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        total_len = len(text)

        while start < total_len:
            end = min(start + self.chunk_size, total_len)

            # Snap to word boundary if not at end of text
            if end < total_len:
                # Look back up to 60 characters for a sentence or word boundary
                boundary = max(
                    text.rfind(". ", start, end),
                    text.rfind("\n", start, end),
                    text.rfind(" ", start, end)
                )
                if boundary > start + (self.chunk_size // 2):
                    end = boundary + 1

            chunk_content = text[start:end].strip()
            if len(chunk_content) >= self.min_chunk_length:
                chunks.append(chunk_content)
            elif not chunks and chunk_content:
                # If it's the only chunk, keep it even if slightly short
                chunks.append(chunk_content)

            if end >= total_len:
                break
            start += step

        return chunks

    def chunk_document(self, document: ParsedDocument) -> List[ChunkRecord]:
        """
        Chunk all pages of a parsed document into structured ChunkRecords.
        Maintains global 0-indexed chunk_index and formatted chunk_id: '{doc_id}_chunk_{index:03d}'.
        """
        chunk_records: List[ChunkRecord] = []
        global_chunk_index = 0

        for page in document.pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            raw_chunks = self._split_text_with_overlap(page_text)
            for chunk_str in raw_chunks:
                # Format: "doc01_chunk_004"
                chunk_id = f"{document.document_id}_chunk_{global_chunk_index:03d}"
                record = ChunkRecord(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    document_name=document.document_name,
                    page_number=page.page_number,
                    text=chunk_str,
                    chunk_index=global_chunk_index,
                    metadata={"page_index": page.page_number - 1}
                )
                chunk_records.append(record)
                global_chunk_index += 1

        return chunk_records

    def chunk_text_blocks(
        self,
        texts: List[str],
        document_id: str = "doc01",
        document_name: str = "document.txt"
    ) -> List[ChunkRecord]:
        """
        Convenience method to chunk raw strings directly with synthetic page numbering.
        """
        records: List[ChunkRecord] = []
        global_index = 0

        for page_num, raw_text in enumerate(texts, start=1):
            splits = self._split_text_with_overlap(raw_text)
            for s in splits:
                chunk_id = f"{document_id}_chunk_{global_index:03d}"
                records.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        document_name=document_name,
                        page_number=page_num,
                        text=s,
                        chunk_index=global_index
                    )
                )
                global_index += 1

        return records
