"""
CLI entry point for the ingestion pipeline.

Invoked as::

    python -m philos.ingest path/to/book.pdf

What it does (end-to-end):
    1. Reads the PDF at the given path.
    2. Converts it to structured Markdown (`loader.load_pdf`).
    3. Splits that Markdown into overlapping chunks (`chunker.chunk_markdown`).
    4. Writes the chunks to `data/processed/<book>.chunks.json`.
    5. Prints a short summary.

This is deliberately thin — it's just glue. All the interesting logic
lives in `loader.py` and `chunker.py`, so you can exercise them from a
Python REPL or a notebook without the CLI in the way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from philos.config import DATA_PROCESSED
from philos.ingest.chunker import chunk_markdown
from philos.ingest.loader import load_pdf


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Define and parse the CLI arguments.

    Kept in its own function so it's easy to unit-test (or call from a
    notebook) without touching `sys.argv`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m philos.ingest",
        description=(
            "Ingest a philosophical text: PDF → Markdown → chunked JSON. "
            "Round 1 only — no embeddings, no LLM calls."
        ),
    )
    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the source PDF (e.g. data/raw/book.pdf).",
    )
    parser.add_argument(
        "--source-name",
        type=str,
        default=None,
        help=(
            "Short slug identifying this book in chunk metadata. "
            "Defaults to the PDF's filename without extension."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_PROCESSED,
        help=(
            "Where to write the chunks JSON. "
            f"Defaults to {DATA_PROCESSED} (from philos.config)."
        ),
    )
    return parser.parse_args(argv)


def run(pdf_path: Path, source_name: str, output_dir: Path) -> Path:
    """Execute the full Round 1 ingestion and return the output path.

    Exposed as a function (not just inlined in `main`) so that later
    rounds and test harnesses can call it directly.
    """
    # Stage 1: extraction. Fails loudly if the file is missing or the
    # PDF contains no extractable text — see loader.py for the checks.
    print(f"[1/3] Loading PDF: {pdf_path}")
    markdown = load_pdf(pdf_path)
    print(f"      → {len(markdown):,} characters of Markdown extracted")

    # Stage 2: chunking. No network, no LLM, no embeddings — purely
    # deterministic text splitting.
    print(f"[2/3] Chunking with source_name={source_name!r}")
    chunks = chunk_markdown(markdown, source_name=source_name)
    print(f"      → {len(chunks):,} chunks produced")

    # Stage 3: persist. We write JSON (not pickle, not a DB) because:
    #   - Human-readable: you can diff two ingest runs, or eyeball
    #     individual chunks, without any tooling.
    #   - Portable: Round 2's embedding step will read this JSON as its
    #     input, keeping ingestion and embedding as separate, resumable
    #     stages.
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source_name}.chunks.json"

    # `indent=2` keeps the file readable; `ensure_ascii=False` preserves
    # Unicode (accents, Greek letters, em-dashes) rather than escaping
    # them — philosophy texts are full of these and the escaped form is
    # painful to eyeball.
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"[3/3] Wrote {output_path}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    """Main entry point. Returns a process exit code."""
    # argparse expects a list; default to sys.argv[1:] when called as a
    # real script, but allow tests to pass a custom list.
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    # Default the source name to the PDF's stem (filename without
    # extension). This is the common case; users rarely need to override.
    source_name = args.source_name or args.pdf_path.stem

    try:
        run(
            pdf_path=args.pdf_path,
            source_name=source_name,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        # Expected, user-facing errors (missing file, empty PDF). Print
        # a clean message to stderr and exit non-zero so shell scripts
        # can detect failure without scraping tracebacks.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    # Pass through the exit code so `python -m philos.ingest` integrates
    # cleanly with shell pipelines and CI.
    raise SystemExit(main())
