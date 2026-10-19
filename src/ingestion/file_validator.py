"""
File Validator Module.

Validates uploaded files (PDF, DOCX, TXT), enforces size limits,
verifies integrity, assigns unique document IDs, and tracks ingestion status.
"""

from __future__ import annotations

import os
import hashlib
import uuid
from enum import Enum
from pathlib import Path
from typing import Optional, Union, BinaryIO
from dataclasses import dataclass, field


class IngestionStatus(str, Enum):
    """Lifecycle stages for document ingestion."""
    PENDING = "PENDING"
    VALIDATED = "VALIDATED"
    PARSING = "PARSING"
    PARSED = "PARSED"
    CHUNKING = "CHUNKING"
    CHUNKED = "CHUNKED"
    EMBEDDING = "EMBEDDING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"


@dataclass
class ValidationResult:
    """Result of file validation."""
    is_valid: bool
    document_id: str
    document_name: str
    file_type: str
    file_size_bytes: int
    error_message: Optional[str] = None
    checksum_sha256: Optional[str] = None


class FileValidator:
    """
    Validates file format, non-emptiness, and size constraints.
    Assigns consistent document IDs based on content hash or UUID.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
    DEFAULT_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

    def __init__(
        self,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE,
        custom_id_prefix: str = "doc"
    ):
        """
        Initialize the file validator.

        Args:
            max_file_size_bytes: Maximum allowed file size in bytes (default 50MB).
            custom_id_prefix: Prefix for generated document IDs.
        """
        self.max_file_size_bytes = max_file_size_bytes
        self.custom_id_prefix = custom_id_prefix

    def generate_document_id(self, content_or_name: Union[bytes, str], counter: Optional[int] = None) -> str:
        """
        Generate a readable, deterministic or sequential document ID.
        If counter is provided, formats as 'doc01', 'doc02', etc.
        Otherwise hashes content for collision-free identifiers.
        """
        if counter is not None:
            return f"{self.custom_id_prefix}{counter:02d}"

        if isinstance(content_or_name, bytes):
            digest = hashlib.sha256(content_or_name).hexdigest()[:8]
            return f"{self.custom_id_prefix}_{digest}"

        clean_name = Path(content_or_name).stem.replace(" ", "_").lower()[:12]
        unique_suffix = uuid.uuid4().hex[:6]
        return f"{self.custom_id_prefix}_{clean_name}_{unique_suffix}"

    def validate_file(
        self,
        file_path_or_buffer: Union[str, Path, BinaryIO],
        filename: Optional[str] = None,
        sequential_counter: Optional[int] = None,
    ) -> ValidationResult:
        """
        Validate file existence, size, extension, and content integrity.

        Args:
            file_path_or_buffer: Path to file or an open binary buffer (e.g. Streamlit UploadedFile).
            filename: Explicit filename when passing a buffer.
            sequential_counter: Optional integer to produce IDs like 'doc01'.

        Returns:
            ValidationResult with status and metadata.
        """
        content_bytes: bytes = b""
        resolved_name: str = ""

        # Case 1: Path string or Path object
        if isinstance(file_path_or_buffer, (str, Path)):
            path = Path(file_path_or_buffer)
            resolved_name = filename or path.name

            if not path.exists():
                return ValidationResult(
                    is_valid=False,
                    document_id="",
                    document_name=resolved_name,
                    file_type="",
                    file_size_bytes=0,
                    error_message=f"File not found at path: {path}"
                )

            try:
                content_bytes = path.read_bytes()
            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    document_id="",
                    document_name=resolved_name,
                    file_type="",
                    file_size_bytes=0,
                    error_message=f"Unable to read file contents: {str(e)}"
                )

        # Case 2: In-memory binary buffer or UploadedFile
        else:
            resolved_name = filename or getattr(file_path_or_buffer, "name", "unknown_file")
            try:
                if hasattr(file_path_or_buffer, "seek"):
                    file_path_or_buffer.seek(0)
                if hasattr(file_path_or_buffer, "read"):
                    content_bytes = file_path_or_buffer.read()
                    if hasattr(file_path_or_buffer, "seek"):
                        file_path_or_buffer.seek(0)
                elif hasattr(file_path_or_buffer, "getvalue"):
                    content_bytes = file_path_or_buffer.getvalue()
                else:
                    content_bytes = bytes(file_path_or_buffer)
            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    document_id="",
                    document_name=resolved_name,
                    file_type="",
                    file_size_bytes=0,
                    error_message=f"Failed to read file buffer: {str(e)}"
                )

        file_size = len(content_bytes)
        ext = Path(resolved_name).suffix.lower()

        # Check 1: Supported extension
        if ext not in self.SUPPORTED_EXTENSIONS:
            return ValidationResult(
                is_valid=False,
                document_id="",
                document_name=resolved_name,
                file_type=ext,
                file_size_bytes=file_size,
                error_message=f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        # Check 2: Non-empty file
        if file_size == 0:
            return ValidationResult(
                is_valid=False,
                document_id="",
                document_name=resolved_name,
                file_type=ext,
                file_size_bytes=0,
                error_message="File is empty (0 bytes). Ingestion rejected."
            )

        # Check 3: Maximum file size limit
        if file_size > self.max_file_size_bytes:
            max_mb = self.max_file_size_bytes / (1024 * 1024)
            current_mb = file_size / (1024 * 1024)
            return ValidationResult(
                is_valid=False,
                document_id="",
                document_name=resolved_name,
                file_type=ext,
                file_size_bytes=file_size,
                error_message=f"File size ({current_mb:.2f} MB) exceeds maximum limit of {max_mb:.2f} MB."
            )

        # Check 4: Format header sanity / magic bytes
        if ext == ".pdf" and not content_bytes.startswith(b"%PDF"):
            return ValidationResult(
                is_valid=False,
                document_id="",
                document_name=resolved_name,
                file_type=ext,
                file_size_bytes=file_size,
                error_message="Corrupted PDF file: Missing standard %PDF header magic bytes."
            )

        if ext == ".docx" and not content_bytes.startswith(b"PK"):
            # DOCX files are zip archives starting with PK (0x50 0x4B)
            return ValidationResult(
                is_valid=False,
                document_id="",
                document_name=resolved_name,
                file_type=ext,
                file_size_bytes=file_size,
                error_message="Corrupted DOCX file: Missing standard ZIP container header."
            )

        checksum = hashlib.sha256(content_bytes).hexdigest()
        doc_id = self.generate_document_id(
            content_bytes if sequential_counter is None else resolved_name,
            counter=sequential_counter
        )

        return ValidationResult(
            is_valid=True,
            document_id=doc_id,
            document_name=resolved_name,
            file_type=ext,
            file_size_bytes=file_size,
            error_message=None,
            checksum_sha256=checksum
        )
