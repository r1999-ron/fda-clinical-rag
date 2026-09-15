# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A RAG (Retrieval-Augmented Generation) pipeline built as a learning/experimentation project over a small set of clinical-trial regulatory PDFs (`data/raw/`). It covers the full pipeline: PDF loading → cleaning → chunking → embedding → vector storage (Qdrant, local/on-disk) → dense retrieval → optional cross-encoder reranking → grounded LLM generation with citation validation. There is also a retrieval evaluation harness that compares dense-only vs. dense+reranked retrieval quality against a hand-labeled question set.

`app.py` is currently empty — there is no unified CLI/entrypoint yet. Each module is run standalone via `if __name__ == "__main__":` blocks for manual inspection during development.

## Setup and commands

This project uses a `uv`-style `.venv` (Python 3.11.9, per `.python-version`) with dependencies in `requirements.txt` (no lockfile, no pyproject.toml).

```bash
# Activate the existing venv
source .venv/bin/activate

# Install/sync dependencies
pip install -r requirements.txt
```

Requires a `.env` file at the project root with `OPENAI_API_KEY` (loaded via `python-dotenv`). Used for both OpenAI embeddings (`text-embedding-3-small`) and generation (`gpt-4.1-mini`).

There is no test framework, linter, or formatter configured — no pytest, no ruff/black config, no CI. `test.py` at the root is an ad hoc manual script, not a real test suite.

### Running individual pipeline stages

Every stage is runnable directly and prints diagnostic output to stdout — this is the primary way to inspect intermediate pipeline state:

```bash
python -m src.ingestion.loader      # load raw PDFs, print first doc/metadata
python -m src.ingestion.chunker     # run structure-aware chunking + inspect headings/chunks
python -m src.embeddings.embedder   # sanity-check embedding output
python -m src.vectorstore.qdrant_store  # build/persist the Qdrant collection from data/raw
python -m src.retrieval.retriever   # run sample queries against the persisted collection
python -m src.generation.generator  # run one question through the full RAG pipeline
```

### Evaluation

```bash
python -m evaluation.evaluate_retrieval
```

Loads `evaluation/questions.json` (hand-labeled expected source/page/terms per question), runs each question through both dense-only and dense+reranked retrieval, and prints Hit@1/3/5 and MRR for both — plus the delta. Relevance is judged by matching expected source filename + expected page number(s) + (optionally) expected terms appearing in the chunk text. When adding new eval questions, all three fields are required for a meaningful judgment (terms are optional but recommended to avoid false positives from page/source matches alone).

`experiments/chunking_results.csv` and `experiments/chunking/` hold prior manual chunking-strategy comparison results — this is where past exploration was recorded, not code to run.

## Architecture

### Data flow

```
data/raw/*.pdf
  → loader.load_pdfs()            (PyPDFLoader, page-level Documents)
  → cleaner.clean_documents()     (strip line-number artifacts, collapse whitespace)
  → chunker.build_chunks()        (pick strategy, see below)
  → qdrant_store.create_vector_store()  (embed + persist to ./qdrant_data)
  ...later, at query time...
  → qdrant_store.load_vector_store()
  → retriever / generator: similarity_search_with_score()
  → [optional] reranker.CrossEncoderReranker.rerank()
  → generator.format_context() + build_prompt()
  → ChatOpenAI.invoke()
  → generator.validate_citations() / get_used_citations()
```

### Chunking strategies (`src/ingestion/chunker.py`)

Three interchangeable strategies exist; `qdrant_store.py` selects one via the `CHUNKING_STRATEGY` constant (`"character"`, `"token"`, or `"structure"`):

- **character** — plain `RecursiveCharacterTextSplitter` on raw character count.
- **token** — same, but sized by `tiktoken` (`cl100k_base`) token count instead of characters.
- **structure** — the most involved strategy. It first runs `split_documents_into_sections()`, which scans each page's lines with `is_heading()` (a heuristic detecting numbered/lettered/roman/uppercase headings while filtering out ToC dotted-leader lines and stray PDF page numbers) to group content under section headings. Section state (`last_section_by_source`) persists **across pages within the same source PDF**, so a section that spans a page break is carried forward correctly. Sections are only then recursively split further if they exceed `max_chunk_size`. This strategy is what produces the `section` field on chunk metadata used later for citations.

All three strategies funnel through `add_chunk_metadata()`, which stamps a sequential `chunk_id` and shifts the 0-indexed PDF `page` into a human-facing `page_number` (`page + 1`).

### Vector store / collection naming (`src/vectorstore/qdrant_store.py`)

Qdrant runs embedded/on-disk (no server) at `./qdrant_data`. Collection names are derived from the active config: `enterprise_{CHUNKING_STRATEGY}_{CHUNK_SIZE}_{CHUNK_OVERLAP}` (e.g. `enterprise_structure_500_50`). This means **changing `CHUNK_SIZE`, `CHUNK_OVERLAP`, or `CHUNKING_STRATEGY` in `qdrant_store.py` targets a different collection**, not the existing one — re-run `create_vector_store` (via the `__main__` block) to build the new collection before querying it, or retrieval/generation/evaluation will fail against a nonexistent collection. Every caller of `load_vector_store()`/`create_vector_store()` is responsible for calling `vector_store.client.close()` (typically in a `finally` block) since Qdrant's local mode holds a file lock (`qdrant_data/.lock`) that blocks concurrent access.

### Generation and citation trust boundary (`src/generation/generator.py`)

The core design principle here: **the LLM only ever emits numeric citation markers like `[1]`, `[2]`; it never sees or produces actual source/page/section metadata.** The backend (`build_citation_map`) independently owns the mapping from citation number → real provenance (source filename, page, section, chunk_id, dense/reranker scores), built directly from the retrieved documents. After generation, `extract_citation_ids()` + `validate_citations()` check that every citation number the model used actually exists in that map — this catches hallucinated citations. `get_used_citations()` then filters the trusted map down to only the citations actually referenced in the final answer, so returned provenance never includes retrieved-but-unused sources.

The reranker step is currently disabled at generation time (`USE_RERANKER = False` in `generate_answer()`) — dense retrieval results are passed straight to generation. It's still exercised in `evaluation/evaluate_retrieval.py` to measure its effect on retrieval metrics.

### Reranking (`src/reranking/reranker.py`)

`CrossEncoderReranker` wraps a `sentence-transformers` `CrossEncoder` (`BAAI/bge-reranker-large`; an alternate smaller model is left commented out). Loading the model is expensive, so instances should be reused across queries rather than reconstructed per call (the evaluation script does this correctly by creating one `reranker` before the loop; be mindful of this when adding new call sites). `rerank()` mutates document metadata in place, stamping both `dense_score` and `reranker_score` before re-sorting and truncating to `top_n`.

## Working with this codebase

- Modules use `src.` absolute imports (e.g. `from src.ingestion.loader import load_pdfs`) and are intended to be run with `python -m` from the project root, not as standalone scripts (`python src/foo/bar.py` will break the imports).
- There's no `__init__.py`-driven package API to keep in sync — each module is fairly self-contained and imported directly by path.
- Diagnostic `print()` statements throughout are intentional and are the current means of manual inspection (no logging framework). Preserve this style when extending existing `__main__` blocks rather than replacing it with logging.
