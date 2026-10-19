"""
Unit Tests for Document Ingestion Pipeline.

Tests file validation, parsing integrity, chunking window logic,
and adherence to the required chunk record schema.
"""

import io
import pytest
from pathlib import Path

from src.ingestion.file_validator import FileValidator, IngestionStatus, ValidationResult
from src.ingestion.document_parser import DocumentParser, ParsedDocument, PageContent, ParsingError
from src.ingestion.chunker import Chunker, ChunkRecord


class TestFileValidator:
    """Tests for file validation, size limits, format checks, and ID generation."""

    def test_valid_txt_file(self):
        validator = FileValidator(max_file_size_bytes=1024 * 1024)
        sample_bytes = b"Hello world, this is a valid document for ingestion."
        result = validator.validate_file(sample_bytes, filename="sample.txt", sequential_counter=1)

        assert result.is_valid is True
        assert result.document_id == "doc01"
        assert result.document_name == "sample.txt"
        assert result.file_type == ".txt"
        assert result.file_size_bytes == len(sample_bytes)
        assert result.error_message is None
        assert result.checksum_sha256 is not None

    def test_unsupported_file_extension(self):
        validator = FileValidator()
        result = validator.validate_file(b"data", filename="table.csv")
        assert result.is_valid is False
        assert "Unsupported file format" in result.error_message

    def test_empty_file_rejected(self):
        validator = FileValidator()
        result = validator.validate_file(b"", filename="empty.txt")
        assert result.is_valid is False
        assert "File is empty" in result.error_message

    def test_file_exceeding_max_size_rejected(self):
        validator = FileValidator(max_file_size_bytes=50)
        large_bytes = b"X" * 100
        result = validator.validate_file(large_bytes, filename="large.txt")
        assert result.is_valid is False
        assert "exceeds maximum limit" in result.error_message

    def test_corrupted_pdf_rejected(self):
        validator = FileValidator()
        # Invalid PDF lacking %PDF header
        bad_pdf = b"NOT_A_REAL_PDF_HEADER_CONTENT"
        result = validator.validate_file(bad_pdf, filename="bad.pdf")
        assert result.is_valid is False
        assert "Corrupted PDF" in result.error_message

    def test_valid_pdf_header_accepted(self):
        validator = FileValidator()
        good_pdf = b"%PDF-1.4 sample pdf content stream"
        result = validator.validate_file(good_pdf, filename="test.pdf")
        assert result.is_valid is True


class TestDocumentParser:
    """Tests for document text extraction and cleaning."""

    def test_parse_txt(self):
        parser = DocumentParser()
        content = "Line 1.\nLine 2.\n\nParagraph 2 with   extra   spaces."
        parsed = parser.parse_txt(content.encode("utf-8"), document_id="doc01", document_name="test.txt")

        assert isinstance(parsed, ParsedDocument)
        assert parsed.document_id == "doc01"
        assert parsed.file_type == ".txt"
        assert len(parsed.pages) >= 1
        assert "Line 1" in parsed.full_text
        assert "Paragraph 2 with extra spaces." in parsed.full_text

    def test_clean_text_preserves_structure(self):
        parser = DocumentParser()
        dirty = "Hello\x00World\r\n\n\n\nNew Paragraph  with  spacing."
        cleaned = parser.clean_text(dirty)
        assert "\x00" not in cleaned
        assert "\n\n\n" not in cleaned
        assert "Hello" in cleaned
        assert "New Paragraph with spacing." in cleaned

    def test_unsupported_parser_dispatch_raises_error(self):
        parser = DocumentParser()
        with pytest.raises(ParsingError):
            parser.parse(b"data", document_id="doc01", document_name="archive.zip")


class TestChunker:
    """Tests for chunking engine and schema compliance."""

    def test_chunk_metadata_schema(self):
        chunker = Chunker(chunk_size=100, chunk_overlap=20)
        parsed_doc = ParsedDocument(
            document_id="doc01",
            document_name="guide.pdf",
            file_type=".pdf",
            pages=[
                PageContent(page_number=5, text="The system retrieves relevant passages from indexed documents.")
            ],
            total_pages=1,
            raw_character_count=65
        )

        chunks = chunker.chunk_document(parsed_doc)
        assert len(chunks) == 1
        chunk = chunks[0]

        # Verify exact schema requirements from specification
        data = chunk.to_dict()
        assert "chunk_id" in data
        assert "document_id" in data
        assert "document_name" in data
        assert "page_number" in data
        assert "text" in data
        assert "chunk_index" in data

        assert chunk.chunk_id == "doc01_chunk_000"
        assert chunk.document_id == "doc01"
        assert chunk.document_name == "guide.pdf"
        assert chunk.page_number == 5
        assert chunk.chunk_index == 0
        assert "The system retrieves" in chunk.text

    def test_sliding_window_overlap(self):
        chunker = Chunker(chunk_size=50, chunk_overlap=15, min_chunk_length=10)
        long_text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau."
        pages = [PageContent(page_number=1, text=long_text)]
        doc = ParsedDocument("doc01", "test.txt", ".txt", pages, 1, len(long_text))

        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1

        # Check successive chunks are properly indexed
        for i, c in enumerate(chunks):
            assert c.chunk_index == i
            assert c.chunk_id == f"doc01_chunk_{i:03d}"

    def test_invalid_overlap_raises_value_error(self):
        with pytest.raises(ValueError):
            Chunker(chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError):
            Chunker(chunk_size=100, chunk_overlap=150)
