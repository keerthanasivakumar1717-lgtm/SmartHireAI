"""
SmartHireAI - Document Loader
Handles multi-format document text extraction (PDF, DOCX, TXT) with robust error handling.
"""

from pathlib import Path
from typing import Union, BinaryIO
import re


class DocumentLoaderError(Exception):
    """Base exception for document loading issues."""
    pass


class UnsupportedFileFormatError(DocumentLoaderError):
    """Raised when an unsupported file extension is provided."""
    pass


class EmptyDocumentError(DocumentLoaderError):
    """Raised when extracted text is empty or purely whitespace."""
    pass


class CorruptedDocumentError(DocumentLoaderError):
    """Raised when parsing fails due to malformed or corrupted file bytes."""
    pass


def sanitize_text(text: str) -> str:
    """Normalize extracted text, strip control characters and normalize whitespace."""
    if not text:
        return ""
    # Strip null characters and unprintable control chars except tabs/newlines
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Replace multiple consecutive spaces with a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Normalize excessive consecutive newlines to at most two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_pdf(source: Union[str, Path, BinaryIO]) -> str:
    """Extract text from a PDF file path or binary stream using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            "The 'pypdf' package is required to load PDF documents. "
            "Please install it with: pip install pypdf"
        )

    try:
        reader = PdfReader(source)
        if len(reader.pages) == 0:
            raise EmptyDocumentError("The uploaded PDF has 0 pages.")

        extracted_pages = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted_pages.append(page_text)

        full_text = "\n".join(extracted_pages)
        if not full_text.strip():
            raise EmptyDocumentError(
                "PDF could be read, but no selectable text was found. "
                "Scanned or image-only PDFs require OCR which is outside current scope."
            )
        return sanitize_text(full_text)
    except EmptyDocumentError:
        raise
    except Exception as exc:
        raise CorruptedDocumentError(f"Failed to read PDF document: {exc}") from exc


def extract_text_from_docx(source: Union[str, Path, BinaryIO]) -> str:
    """Extract text from a DOCX file path or binary stream using python-docx."""
    try:
        import docx
    except ImportError:
        raise ImportError(
            "The 'python-docx' package is required to load DOCX documents. "
            "Please install it with: pip install python-docx"
        )

    try:
        doc = docx.Document(source)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        # Also extract table text if present
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragraphs.append(row_text)

        full_text = "\n".join(paragraphs)
        if not full_text.strip():
            raise EmptyDocumentError("The DOCX document contains no text.")
        return sanitize_text(full_text)
    except EmptyDocumentError:
        raise
    except Exception as exc:
        raise CorruptedDocumentError(f"Failed to read DOCX document: {exc}") from exc


def extract_text_from_txt(source: Union[str, Path, BinaryIO]) -> str:
    """Extract text from a plain text file path or stream."""
    try:
        if isinstance(source, (str, Path)):
            with open(source, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        elif hasattr(source, "read"):
            raw = source.read()
            if isinstance(raw, bytes):
                content = raw.decode("utf-8", errors="replace")
            else:
                content = str(raw)
        else:
            raise ValueError("Unsupported source type for text file.")

        if not content.strip():
            raise EmptyDocumentError("The text document is empty.")
        return sanitize_text(content)
    except EmptyDocumentError:
        raise
    except Exception as exc:
        raise CorruptedDocumentError(f"Failed to read text document: {exc}") from exc


def load_resume_document(
    source: Union[str, Path, BinaryIO], filename_hint: str = ""
) -> str:
    """
    Unified entry point to extract text from an uploaded resume.
    Accepts a filepath (str or Path) or a binary file stream (such as Streamlit UploadedFile).
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        ext = path.suffix.lower()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")
    else:
        # Source is a file-like stream
        name = getattr(source, "name", "") or filename_hint
        ext = Path(name).suffix.lower() if name else ""

    if ext == ".pdf":
        return extract_text_from_pdf(source)
    elif ext in [".docx", ".doc"]:
        return extract_text_from_docx(source)
    elif ext in [".txt", ".text", ".md"]:
        return extract_text_from_txt(source)
    else:
        raise UnsupportedFileFormatError(
            f"Unsupported file format '{ext}'. SmartHireAI supports PDF (.pdf), Word (.docx), and Text (.txt) resumes."
        )

