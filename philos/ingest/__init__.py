"""
philos.ingest — the ingestion pipeline.

In Round 1 this sub-package contains only two stages:

    PDF file  ──(loader.load_pdf)──►  Markdown string
    Markdown  ──(chunker.chunk_markdown)──►  list[dict] of chunks

The CLI in `__main__.py` stitches them together.

Future rounds will add more stages to this sub-package:
  - metadata enrichment (LLM labels each chunk's argument role)
  - embedding + ChromaDB persistence
  - BM25 index construction

Each will be a new module here so the pipeline stays readable as a
linear sequence of transformations.
"""

from philos.ingest.chunker import chunk_markdown
from philos.ingest.loader import load_pdf

# Public API of this sub-package. Keeping this list short and explicit
# makes it obvious what "stages" exist today — and easy to spot when a
# new round adds another one.
__all__ = ["load_pdf", "chunk_markdown"]
