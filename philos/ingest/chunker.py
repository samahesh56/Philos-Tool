"""
Markdown → chunks.

Stage 2 of the ingestion pipeline.

What a "chunk" is here:
  A contiguous slice of the source text, small enough to embed usefully
  and large enough to carry argumentative context. Each chunk will
  eventually get its own vector in ChromaDB and its own row in the BM25
  index (Round 2 & 3).

Why SentenceSplitter (from LlamaIndex)?
  It's a *recursive* splitter: it first tries to break on paragraph
  boundaries, then sentences, only falling back to hard character cuts
  when a run of prose is genuinely longer than CHUNK_SIZE. That means
  chunks almost never start or end mid-sentence — a property we care
  about because a half-sentence is a bad retrieval unit and a worse
  citation.

What's NOT in this module (deferred to later rounds):
  - No LLM-derived metadata (`argument_role`, `context_summary`). Those
    require an API call per chunk; we'll add them in Round 2 so we can
    keep Round 1 runnable with zero API keys.
  - No embeddings. SentenceSplitter gives us *character* offsets for
    free, not token counts — that's fine for inspection but the
    embedding pipeline (Round 2) will introduce a proper tokenizer.
"""

from __future__ import annotations

import hashlib

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

from philos.config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_markdown(markdown: str, source_name: str) -> list[dict]:
    """Split a Markdown document into overlapping chunks.

    Parameters
    ----------
    markdown:
        The full document text, typically the output of `load_pdf`.
    source_name:
        Short, human-readable identifier for the book (e.g.
        ``"existentialism-is-a-humanism"``). Stored on every chunk so
        later stages can answer "which book did this chunk come from?"

    Returns
    -------
    list[dict]
        One dict per chunk, in reading order. Schema:

            {
                "id":         str,   # stable hash of source_name + index + text
                "source":     str,   # echoes the source_name argument
                "index":      int,   # 0-based position in the book
                "text":       str,   # the chunk's prose
                "char_start": int,   # offset into the original markdown
                "char_end":   int,   # exclusive end offset
            }

        This is intentionally a plain list-of-dicts (not a custom class)
        so it serializes straight to JSON for inspection.
    """
    # LlamaIndex's splitter operates on Document objects, not raw
    # strings. Wrapping is a one-liner. The `text=` kwarg is the only
    # field we care about at this stage; we'll start attaching real
    # metadata to Documents in Round 2.
    doc = Document(text=markdown)

    # Configure the splitter with our tuned defaults from config.py.
    # Key behaviors:
    #   - chunk_size is measured in *tokens* (SentenceSplitter's default
    #     tokenizer is a simple whitespace/punct splitter — close enough
    #     for sizing purposes; we'll switch to a model-specific tokenizer
    #     in Round 2 when it starts mattering for embeddings).
    #   - chunk_overlap ensures cross-boundary context survives (see the
    #     CHUNK_OVERLAP comment in config.py for the "why").
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # `get_nodes_from_documents` returns a list of TextNode objects in
    # reading order. Each node carries its text plus character offsets
    # back into the source — which we want for both debugging ("what
    # page did this come from?") and future citation work.
    nodes = splitter.get_nodes_from_documents([doc])

    chunks: list[dict] = []
    for index, node in enumerate(nodes):
        text = node.get_content()

        # A stable, content-addressed ID. Using a hash (rather than a
        # plain counter) means:
        #   - The same chunk text produces the same ID across re-runs,
        #     so Round 2's embedding step can tell "did this chunk
        #     actually change, or is it the same prose?" and skip
        #     re-embedding unchanged chunks.
        #   - Including `source_name` and `index` in the hash prevents
        #     collisions between identical boilerplate (e.g. headers)
        #     across different books.
        #
        # We use SHA-1 truncated to 16 hex chars — plenty of entropy for
        # a single book's worth of chunks, and compact in JSON.
        id_input = f"{source_name}|{index}|{text}".encode("utf-8")
        chunk_id = hashlib.sha1(id_input).hexdigest()[:16]

        # Character offsets come from the node's `start_char_idx` /
        # `end_char_idx` attributes. LlamaIndex populates these for
        # SentenceSplitter outputs. They're useful for:
        #   - Debugging ("show me the raw source around chunk 42")
        #   - Future work on precise citations (mapping back to page #s
        #     via the page markers PyMuPDF4LLM left in the Markdown)
        char_start = node.start_char_idx or 0
        char_end = node.end_char_idx or (char_start + len(text))

        chunks.append(
            {
                "id": chunk_id,
                "source": source_name,
                "index": index,
                "text": text,
                "char_start": char_start,
                "char_end": char_end,
            }
        )

    return chunks
