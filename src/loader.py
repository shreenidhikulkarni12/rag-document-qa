"""
Loads text out of uploaded files. Supports .txt and .pdf for now -
easy to extend to .docx later using the same pattern.
"""
from pypdf import PdfReader
import io


def load_text(filename: str, file_bytes: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if filename.lower().endswith(".txt"):
        return file_bytes.decode("utf-8")

    raise ValueError(f"Unsupported file type: {filename}")
