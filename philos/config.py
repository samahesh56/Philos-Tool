"""
Central configuration constants for Philos.

Why this file exists:
  The blueprint sets chunking parameters (size, overlap) and data-directory
  conventions that are referenced from multiple places. Centralizing them
  here means we change them in exactly ONE spot when we tune the system.

Everything in here is pure data — no I/O, no side effects — so importing
this module is cheap and safe from anywhere in the codebase.
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Chunking parameters
# ──────────────────────────────────────────────────────────────────────
# The blueprint recommends 512–1024 tokens per chunk with ~15% overlap.
# Reasoning behind the range:
#   - Too small (<512): loses argumentative context; a single premise gets
#     split from its conclusion, breaking retrieval of dense philosophical
#     reasoning.
#   - Too large (>1024): dilutes embedding signal; the chunk's vector
#     averages over too many distinct ideas and stops being discriminative.
# We pick the midpoint (768) as a reasonable default for dense prose.
CHUNK_SIZE: int = 768

# 15% of 768 ≈ 115. Overlap exists so that a sentence straddling a chunk
# boundary still appears (in full) in at least one chunk. Without overlap,
# retrieval misses arguments that happen to cross the split point.
CHUNK_OVERLAP: int = 115

# ──────────────────────────────────────────────────────────────────────
# Filesystem layout
# ──────────────────────────────────────────────────────────────────────
# All paths are resolved relative to the repo root (the parent of the
# `philos/` package directory). This lets the CLI work no matter what
# working directory the user runs it from, as long as they run it from
# somewhere inside the repo.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Where the user drops source PDFs (gitignored — may be copyrighted).
DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"

# Where the ingest pipeline writes its outputs (gitignored — regenerable).
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
