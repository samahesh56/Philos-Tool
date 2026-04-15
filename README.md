# Philos

A RAG-powered reading companion for a single philosophical text. Philos
surfaces relevant passages, asks Socratic questions, and grounds every
claim in the source material — so you read the philosopher, not a
summary of the philosopher.

See the project prompt for the full architectural blueprint and
pedagogical principles.

## Status

Round 1 complete: **ingestion pipeline** (PDF → chunked JSON). No
embeddings, retrieval, generation, or UI yet — those arrive in later
rounds.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No API keys are required for Round 1.

## Usage (Round 1)

1. Drop a PDF into `data/raw/`. For the initial test we're using
   *Existentialism is a Humanism* (Sartre) — short enough to iterate on
   the full pipeline in seconds.
2. Run the ingester:

   ```bash
   python -m philos.ingest data/raw/existentialism-is-a-humanism.pdf
   ```

3. Inspect the output at
   `data/processed/existentialism-is-a-humanism.chunks.json`. You should
   see a JSON array of chunks, each with `id`, `text`, `source`,
   `index`, and character offsets.

### What to look for when eyeballing the output

- **Chunk count**: a ~50-page book at ~768 tokens/chunk should yield
  roughly 30–80 chunks.
- **Boundaries**: chunks should start and end on sentence boundaries,
  not mid-word.
- **Overlap**: adjacent chunks should share ~15% of their content —
  that's the sliding-window context guard.
- **Structure preserved**: section headings from the book should appear
  as Markdown headings (`##`, `###`) inside chunks.

## Project layout

```
Philos-Tool/
├── philos/                     # the Python package
│   ├── config.py               # chunk size, overlap, paths
│   └── ingest/                 # Round 1: PDF → Markdown → chunks
│       ├── loader.py
│       ├── chunker.py
│       └── __main__.py         # `python -m philos.ingest ...`
├── data/
│   ├── raw/                    # your source PDFs (gitignored)
│   └── processed/              # chunked JSON outputs (gitignored)
├── requirements.txt
├── .env.example                # placeholder for future API keys
└── .gitignore
```

## Roadmap

- **Round 1 (done)**: Ingestion — PDF extraction + chunking.
- **Round 2**: Metadata enrichment (LLM tags `argument_role`,
  `context_summary` per chunk) + OpenAI embeddings + ChromaDB.
- **Round 3**: BM25 index + hybrid retrieval with Reciprocal Rank Fusion.
- **Round 4**: Cross-encoder reranking + Socratic system prompt +
  generation loop (Claude Sonnet).
- **Round 5**: Conversation memory + Streamlit chat UI.
- **Round 6**: RAGAS evaluation harness + golden question set.
