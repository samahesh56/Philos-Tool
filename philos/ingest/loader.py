"""
PDF → Markdown loader.

Stage 1 of the ingestion pipeline.

Why Markdown (not plain text)?
  Philosophy books have deep structure — parts, chapters, sections,
  footnotes, page numbers — and that structure is the scaffolding for
  citations like "(Part II, Ch. 1, p. 73)". If we flatten the PDF to
  plain text we lose the ability to cite accurately. PyMuPDF4LLM emits
  Markdown that preserves:
    - heading levels (# / ## / ###) from the PDF's table of contents
    - page breaks (as horizontal rules)
    - inline structure like emphasis and lists

Why PyMuPDF4LLM specifically?
  - CPU-only, fast, no GPU needed.
  - Output is explicitly tuned for downstream LLM consumption (hence
    the "4LLM" in the name).
  - Handles most modern PDFs out of the box.

Known limitation:
  Scanned/image-only PDFs produce empty strings — there's no OCR layer
  to read. We surface that as a clear error rather than silently
  writing zero chunks.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf4llm


def load_pdf(pdf_path: str | Path) -> str:
    """Extract a PDF's text as structured Markdown.

    Parameters
    ----------
    pdf_path:
        Path to a `.pdf` file. Accepts either a string or a `Path`
        object so callers don't have to convert.

    Returns
    -------
    str
        A single Markdown document. Headings, page markers, and basic
        inline structure are preserved.

    Raises
    ------
    FileNotFoundError
        If `pdf_path` does not exist on disk.
    ValueError
        If the PDF yields no extractable text (likely a scanned image
        PDF with no OCR layer).
    """
    # Normalize input to a Path so we can use filesystem methods below
    # without worrying about whether the caller passed a string.
    path = Path(pdf_path)

    if not path.exists():
        # Clearer than the default FileNotFoundError from pymupdf deep
        # inside a native call stack — we want the user to know which
        # file is missing, in our terms.
        raise FileNotFoundError(f"PDF not found: {path}")

    # The actual extraction. pymupdf4llm opens the document, walks every
    # page, and stitches together a Markdown string. For a ~50-page book
    # this takes well under a second on a laptop.
    #
    # Note: pymupdf4llm accepts a string path, not a Path object, so we
    # convert explicitly.
    markdown: str = pymupdf4llm.to_markdown(str(path))

    # Guard against silent failures. A scanned PDF with no OCR will
    # return an empty or whitespace-only string. That would propagate
    # through chunking as "zero chunks", which is confusing. Fail loud
    # instead, so the user knows to run OCR first.
    if not markdown.strip():
        raise ValueError(
            f"No extractable text in {path}. "
            "This is likely a scanned PDF — run OCR (e.g. ocrmypdf) "
            "before ingesting."
        )

    return markdown
