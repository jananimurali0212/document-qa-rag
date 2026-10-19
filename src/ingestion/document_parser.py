"""
Document Parser Module.

Extracts text from PDF (using PyMuPDF), DOCX (using python-docx), and TXT files,
tracking page numbers and metadata while handling malformed content safely.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, BinaryIO
from dataclasses import dataclass, field


class ParsingError(Exception):
    """Exception raised when document parsing fails."""
    pass


@dataclass
class PageContent:
    """Represents text extracted from a specific page/section."""
    page_number: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """Complete parsed document with ordered pages and metadata."""
    document_id: str
    document_name: str
    file_type: str
    pages: List[PageContent]
    total_pages: int
    raw_character_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Return the combined text of all pages."""
        return "\n\n".join(page.text for page in self.pages)


class DocumentParser:
    """
    Modular parser for PDF, DOCX, and TXT files.
    Preserves page boundaries for precise citation grounding.
    """

    def __init__(self, clean_whitespace: bool = True):
        self.clean_whitespace = clean_whitespace

    def clean_text(self, text: str) -> str:
        """Clean extracted text while preserving meaningful paragraphs."""
        if not text:
            return ""
        # Remove null bytes and non-printable control chars except \n, \t, \r
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
        if self.clean_whitespace:
            # Replace multiple spaces/tabs with single space
            text = re.sub(r"[ \t]+", " ", text)
            # Normalize excessive newlines to double newline
            text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def parse_pdf(
        self,
        file_input: Union[bytes, BinaryIO, str, Path],
        document_id: str,
        document_name: str
    ) -> ParsedDocument:
        """
        Extract text from PDF pages using PyMuPDF (fitz).
        """
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise ParsingError("PyMuPDF (fitz) is not installed. Please install it with `pip install pymupdf`.") from e

        pages: List[PageContent] = []
        doc = None
        try:
            if isinstance(file_input, (str, Path)):
                doc = fitz.open(str(file_input))
            elif isinstance(file_input, bytes):
                doc = fitz.open(stream=file_input, filetype="pdf")
            elif hasattr(file_input, "read"):
                if hasattr(file_input, "seek"):
                    file_input.seek(0)
                data = file_input.read()
                doc = fitz.open(stream=data, filetype="pdf")
            else:
                raise ParsingError(f"Unsupported input type for PDF: {type(file_input)}")

            total_pages = len(doc)
            for page_index in range(total_pages):
                page = doc.load_page(page_index)
                page_text = page.get_text("text") or ""
                cleaned = self.clean_text(page_text)
                if cleaned:
                    pages.append(
                        PageContent(
                            page_number=page_index + 1,
                            text=cleaned,
                            metadata={"page_index": page_index}
                        )
                    )

            if not pages:
                # Still record empty document representation with 1 empty page to allow valid structure
                pages.append(PageContent(page_number=1, text=""))

            total_chars = sum(len(p.text) for p in pages)
            return ParsedDocument(
                document_id=document_id,
                document_name=document_name,
                file_type=".pdf",
                pages=pages,
                total_pages=total_pages,
                raw_character_count=total_chars,
                metadata={"title": doc.metadata.get("title", ""), "author": doc.metadata.get("author", "")}
            )
        except Exception as e:
            raise ParsingError(f"Failed to parse PDF document '{document_name}': {str(e)}") from e
        finally:
            if doc is not None:
                doc.close()

    def parse_docx(
        self,
        file_input: Union[bytes, BinaryIO, str, Path],
        document_id: str,
        document_name: str
    ) -> ParsedDocument:
        """
        Extract text from Microsoft Word (.docx) files using python-docx.
        """
        try:
            from docx import Document
        except ImportError as e:
            raise ParsingError("python-docx is not installed. Please install it with `pip install python-docx`.") from e

        try:
            if isinstance(file_input, (str, Path)):
                doc = Document(str(file_input))
            elif isinstance(file_input, bytes):
                doc = Document(io.BytesIO(file_input))
            elif hasattr(file_input, "read"):
                if hasattr(file_input, "seek"):
                    file_input.seek(0)
                doc = Document(file_input)
            else:
                raise ParsingError(f"Unsupported input type for DOCX: {type(file_input)}")

            paragraphs: List[str] = []
            for p in doc.paragraphs:
                text = p.text.strip()
                if text:
                    paragraphs.append(text)

            # Also extract text inside tables
            for table in doc.tables:
                for row in table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_cells:
                        paragraphs.append(" | ".join(row_cells))

            full_text = "\n\n".join(paragraphs)
            cleaned_text = self.clean_text(full_text)

            # DOCX does not natively store printed page numbers without a layout engine.
            # We partition by logical sections/paragraphs or 3000 chars per virtual page to preserve page tracking.
            virtual_pages: List[PageContent] = []
            chars_per_virtual_page = 3000

            if cleaned_text:
                chunks_raw = [
                    cleaned_text[i:i + chars_per_virtual_page]
                    for i in range(0, len(cleaned_text), chars_per_virtual_page)
                ]
                for idx, chunk_str in enumerate(chunks_raw):
                    virtual_pages.append(
                        PageContent(
                            page_number=idx + 1,
                            text=chunk_str.strip(),
                            metadata={"virtual_page": True}
                        )
                    )
            else:
                virtual_pages.append(PageContent(page_number=1, text=""))

            return ParsedDocument(
                document_id=document_id,
                document_name=document_name,
                file_type=".docx",
                pages=virtual_pages,
                total_pages=len(virtual_pages),
                raw_character_count=len(cleaned_text),
                metadata={"num_paragraphs": len(paragraphs)}
            )
        except Exception as e:
            raise ParsingError(f"Failed to parse DOCX document '{document_name}': {str(e)}") from e

    def parse_txt(
        self,
        file_input: Union[bytes, BinaryIO, str, Path],
        document_id: str,
        document_name: str
    ) -> ParsedDocument:
        """
        Extract text from plain text (.txt) files with encoding auto-detection.
        """
        raw_bytes: bytes = b""
        try:
            if isinstance(file_input, (str, Path)):
                raw_bytes = Path(file_input).read_bytes()
            elif isinstance(file_input, bytes):
                raw_bytes = file_input
            elif hasattr(file_input, "read"):
                if hasattr(file_input, "seek"):
                    file_input.seek(0)
                raw_bytes = file_input.read()
            else:
                raise ParsingError(f"Unsupported input type for TXT: {type(file_input)}")

            # Try utf-8 first, fallback to latin-1
            text = ""
            for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
                try:
                    text = raw_bytes.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue

            cleaned = self.clean_text(text)
            chars_per_virtual_page = 3000
            pages: List[PageContent] = []
            if cleaned:
                splits = [
                    cleaned[i:i + chars_per_virtual_page]
                    for i in range(0, len(cleaned), chars_per_virtual_page)
                ]
                for idx, piece in enumerate(splits):
                    pages.append(PageContent(page_number=idx + 1, text=piece.strip()))
            else:
                pages.append(PageContent(page_number=1, text=""))

            return ParsedDocument(
                document_id=document_id,
                document_name=document_name,
                file_type=".txt",
                pages=pages,
                total_pages=len(pages),
                raw_character_count=len(cleaned),
                metadata={}
            )
        except Exception as e:
            raise ParsingError(f"Failed to parse TXT document '{document_name}': {str(e)}") from e

    def parse(
        self,
        file_input: Union[bytes, BinaryIO, str, Path],
        document_id: str,
        document_name: str
    ) -> ParsedDocument:
        """
        Auto-detect file extension and dispatch to the corresponding parser.
        """
        ext = Path(document_name).suffix.lower()
        if ext == ".pdf":
            return self.parse_pdf(file_input, document_id, document_name)
        elif ext == ".docx":
            return self.parse_docx(file_input, document_id, document_name)
        elif ext == ".txt":
            return self.parse_txt(file_input, document_id, document_name)
        else:
            raise ParsingError(f"Unsupported extension '{ext}' for file '{document_name}'")
