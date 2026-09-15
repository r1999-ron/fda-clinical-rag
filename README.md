# Enterprise Regulatory RAG

An enterprise-style Retrieval-Augmented Generation (RAG) system for grounded question answering over clinical and regulatory guidance documents.

The project focuses on more than basic vector search. It explores **document ingestion, structure-aware chunking, incremental indexing, Dense and BM25 retrieval, query-aware routing, conversational query rewriting, grounded generation, citation validation, safe abstention, observability, and systematic RAG evaluation**.

---

## Overview

The system currently operates over:

| Metric | Value |
|---|---:|
| Documents | 15 PDFs |
| Pages | 361 |
| Indexed chunks | 2,534 |
| Chunking policy | Structure-aware, max 500 / overlap 50 |
| Vector store | Qdrant |
| Embeddings | OpenAI `text-embedding-3-small` |
| Generation | OpenAI LLM |
| API | FastAPI |
| Lexical retrieval | BM25 |

The corpus covers areas including:

- Clinical trial guidance
- Adverse event reporting
- Drug safety
- Good Clinical Practice
- Informed consent
- Trial protocols
- Investigator responsibilities
- Sponsor responsibilities
- IRB guidance
- Safety monitoring
- Clinical data integrity
- Postmarketing safety reporting
- Research subject protections
- Electronic records
- Clinical trial reporting

---

# Why This Project?

A basic RAG implementation often looks like:

```text
Documents
   ↓
Chunk
   ↓
Embed
   ↓
Vector Search
   ↓
LLM
```

That works for many semantic questions, but regulatory document search introduces additional problems:

- Exact regulatory identifiers are difficult for semantic retrieval.
- Relevant information may be split across chunks or footnotes.
- Users may ask conversational follow-up questions.
- Generated answers need verifiable provenance.
- The system should abstain rather than answer from model memory.
- Retrieval quality needs to be evaluated independently from generation quality.
- Indexing should support document updates without rebuilding everything.

This project explores those problems explicitly.

---

# High-Level Architecture

The system separates **offline indexing** from **online query serving**.

```text
                        OFFLINE INDEXING

                    Regulatory PDFs
                           │
                           ▼
                  SHA-256 / Manifest
                           │
                           ▼
                     Load + Clean
                           │
                           ▼
               Structure-Aware Chunking
                    max 500 / overlap 50
                           │
                           ▼
               Metadata + Deterministic IDs
                           │
                           ▼
                       Embeddings
                           │
                           ▼
                         Qdrant


                         ONLINE QUERY

                         User Query
                             │
                             ▼
                      Session History
                             │
                             ▼
                 Conditional Query Rewrite
                             │
                             ▼
                    Metadata Filtering
                             │
                             ▼
                     Retrieval Router
                        /          \
                       /            \
               Semantic Query    Exact CFR
                    │                │
                  Dense             BM25
                    │                │
                  Qdrant       Exact Anchor
                    │                │
                    │        Context Expansion
                    │                │
                    └───────┬────────┘
                            ▼
                      Top-K Evidence
                            │
                            ▼
                      Grounded LLM
                            │
                            ▼
                    Citation Processing
                            │
                            ▼
                     Answer / Abstain
                            │
                            ▼
                      Request Trace
```

---

# Project Structure

```text
rag-deep-dive/
│
├── data/
│   └── raw/
│       └── *.pdf
│
├── evaluation/
│   ├── evaluate_retrieval.py
│   ├── evaluate_expanded_retrieval.py
│   ├── evaluate_generation.py
│   ├── questions.json
│   ├── expanded_retrieval_questions.json
│   └── generation_questions.json
│
├── src/
│   ├── api/
│   │   ├── app.py
│   │   ├── models.py
│   │   └── session_store.py
│   │
│   ├── conversation/
│   │   ├── conversational_rag.py
│   │   ├── models.py
│   │   └── query_rewriter.py
│   │
│   ├── embeddings/
│   │   └── embedder.py
│   │
│   ├── generation/
│   │   └── generator.py
│   │
│   ├── indexing/
│   │   ├── incremental_indexer.py
│   │   └── manifest.py
│   │
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── cleaner.py
│   │   └── chunker.py
│   │
│   ├── observability/
│   │   └── trace.py
│   │
│   ├── reranking/
│   │   └── reranker.py
│   │
│   ├── retrieval/
│   │   ├── retriever.py
│   │   ├── filtered_retriever.py
│   │   └── hybrid_retriever.py
│   │
│   └── vectorstore/
│       └── qdrant_store.py
│
├── ui/
│   └── index.html
│
├── requirements.txt
└── README.md
```

---

# 1. Document Ingestion

Documents are loaded page-by-page while retaining metadata required for retrieval and provenance.

Important metadata includes:

```text
document_id
source
page_number
section
chunk_id
```

Example deterministic chunk ID:

```text
05_informed_consent_guidance_page_10_chunk_11
```

Deterministic chunk identity is useful across:

- Dense retrieval
- BM25 retrieval
- Filtering
- Citations
- Evaluation
- Debugging
- Context expansion

---

# 2. Incremental Indexing

The indexing pipeline avoids rebuilding the complete knowledge base whenever one document changes.

Each PDF is tracked through a content fingerprint/manifest.

Documents can be classified as:

```text
NEW
    → process and index

UPDATED
    → replace existing indexed content

UNCHANGED
    → skip

REMOVED
    → remove indexed content
```

Run:

```bash
python -m src.indexing.incremental_indexer
```

A subsequent run with unchanged documents should produce output similar to:

```text
Found 15 PDFs.

SKIP: 01_clinical_trial_guidance.pdf is unchanged.
SKIP: 02_adverse_event_reporting.pdf is unchanged.
...
SKIP: 15_clinical_trial_reporting.pdf is unchanged.

Incremental indexing complete.
```

This prevents unnecessary document processing and embedding calls.

---

# 3. Structure-Aware Chunking

The final ingestion pipeline uses a structure-aware chunking strategy with:

```text
max_size = 500 characters
overlap  = 50 characters
```

Where possible, document structure such as sections and page metadata is preserved.

Overlap helps reduce information loss around chunk boundaries.

For example, without overlap:

```text
Chunk 1:
A participant may withdraw from the

Chunk 2:
clinical trial without penalty...
```

With overlap, relevant boundary information has a better chance of remaining together.

---

# 4. Chunking Evaluation

Chunk size was treated as a retrieval hyperparameter rather than selected arbitrarily.

Several configurations were evaluated using the same retrieval benchmark.

| Configuration | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| **500 / 50** | **80.00%** | **100.00%** | **100.00%** | **0.9000** |
| 800 / 100 | 80.00% | 100.00% | 100.00% | 0.8889 |
| 1200 / 150 | 73.33% | 93.33% | 93.33% | 0.8333 |
| 300 / 50 | 73.33% | 86.67% | 86.67% | 0.8000 |
| Structure-aware 800 / 100 | 73.33% | 100.00% | 100.00% | 0.8556 |

The 500/50 configuration provided the strongest initial MRR while maintaining full Hit@3 and Hit@5 coverage.

Run the initial retrieval/chunking evaluation with:

```bash
python -m evaluation.evaluate_retrieval
```

---

# 5. Dense Semantic Retrieval

Dense retrieval is the default strategy for natural-language semantic questions.

Example:

```text
What is a master protocol?
```

Flow:

```text
Query
   ↓
Embedding
   ↓
Qdrant similarity search
   ↓
Top-K evidence
```

Dense retrieval allows semantically similar text to match even when the wording differs.

Example:

```text
Query:
Can the participant leave the study?

Document:
The subject may discontinue participation.
```

The wording differs, but the meaning is similar.

---

# 6. BM25 Lexical Retrieval

Dense retrieval is less reliable when exact identifiers matter.

Consider:

```text
21 CFR 50.25(a)(2)
```

versus:

```text
21 CFR 50.25(a)(6)
```

These references are structurally and semantically similar, but they refer to different regulatory provisions.

BM25 provides lexical retrieval where exact terms receive strong ranking signals.

The BM25 corpus is built from the same canonical chunks used by Dense retrieval.

Conceptually:

```text
Canonical chunks
      ↓
Tokenization
      ↓
BM25 index
      ↓
Query tokens
      ↓
BM25 scoring
      ↓
Ranked chunks
```

For the current 2,534-chunk prototype, BM25 is maintained in memory.

For larger production deployments, a persisted lexical engine such as OpenSearch or Elasticsearch would be more appropriate.

---

# 7. Query-Aware Retrieval Routing

Evaluation showed that no single retrieval algorithm performed best for every query type.

The final retrieval policy is:

```text
Exact CFR section/subsection
        ↓
       BM25

Everything else
        ↓
       Dense
```

Conceptually:

```python
def get_retrieval_strategy(query: str) -> str:

    if has_regulatory_identifier(query):
        return "bm25"

    return "dense"
```

Examples:

```text
"What is a master protocol?"
        → Dense
```

```text
"What does 21 CFR 50.25(a)(2) require?"
        → BM25
```

A broad regulatory question remains semantic:

```text
"What requirements are covered by 21 CFR Part 11?"
        → Dense
```

The router therefore does not simply check whether `"CFR"` appears in the query.

Exact section/subsection syntax is detected using deterministic pattern matching.

---

# 8. Local Context Expansion

Exact lexical retrieval exposed another important failure mode.

For:

```text
What does 21 CFR 50.25(a)(2) require?
```

BM25 correctly found the exact regulatory reference, but the relevant PDF contained:

```text
Chunk 8:
Reasonably Foreseeable Risks and Discomforts [20]

Chunk 9:
...

Chunk 10:
...

Chunk 11:
20 → 21 CFR 50.25(a)(2)
```

The exact identifier existed in a footnote, while the substantive explanation appeared several chunks earlier.

Therefore, for exact regulatory lookups, BM25 is used as an **anchor locator**.

```text
Exact CFR query
      ↓
BM25
      ↓
Strongest lexical anchor
      ↓
Neighbor context expansion
      ↓
Restore original document order
      ↓
Generation evidence
```

The current strategy expands local context around the anchor instead of globally increasing chunk size.

This preserves the retrieval benefits of the 500/50 baseline while addressing document-structure-specific failures.

---

# 9. Hybrid Retrieval Experiment

Dense + BM25 were also evaluated using Reciprocal Rank Fusion (RRF).

Conceptually:

```text
Dense ranking ────┐
                  ├── RRF → fused ranking
BM25 ranking ─────┘
```

RRF rewards results that appear high across multiple ranked lists.

However, evaluation showed that Hybrid retrieval was **not universally superior**.

For exact regulatory identifiers, cross-retriever agreement could sometimes compete with a stronger exact lexical result.

This motivated query-aware routing instead of applying Hybrid retrieval globally.

---

# 10. Final Retrieval Evaluation

The expanded benchmark compared:

- Dense
- BM25
- Hybrid RRF
- Routed retrieval

| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| Dense | 64.10% | 79.49% | 82.05% | 0.7115 |
| BM25 | 64.10% | 82.05% | 82.05% | 0.7313 |
| Hybrid RRF | 61.54% | 79.49% | 87.18% | 0.7201 |
| **Routed** | **71.79%** | **89.74%** | **92.31%** | **0.7970** |

Run:

```bash
python -m evaluation.evaluate_expanded_retrieval
```

### Metric interpretation

**Hit@K**

> Does at least one relevant piece of evidence appear within the first K results?

**MRR**

> How early does the first relevant result appear?

Therefore:

```text
Hit@K → retrieval coverage
MRR   → ranking quality
```

---

# 11. Conversational Query Rewriting

The system supports multi-turn questions.

Example:

```text
User:
What risks must be explained during informed consent?

User:
What about alternatives?
```

The second query is understandable conversationally but weak for standalone retrieval.

An LLM-based query rewriter converts it into something similar to:

```text
What alternative procedures or treatments should be explained
during informed consent?
```

Flow:

```text
Conversation History
        +
Current Question
        ↓
LLM Query Rewriter
        ↓
Standalone Retrieval Query
        ↓
Retrieval Router
```

Already-standalone questions skip rewriting.

Example:

```text
What is a master protocol?
```

produces:

```text
query_rewritten = false
```

This avoids unnecessary model calls and reduces the risk of query-intent drift.

---

# 12. Metadata Filtering

Requests can restrict retrieval to a particular document.

Example:

```json
{
  "session_id": "filter-demo",
  "question": "What risks must be explained during informed consent?",
  "filters": {
    "document_id": "05_informed_consent_guidance"
  }
}
```

The filter is applied **before generation**.

Conceptually:

```text
User / Authorization Scope
          ↓
Metadata Filter
          ↓
Retriever
          ↓
Allowed Evidence Only
          ↓
LLM
```

Dense retrieval uses Qdrant metadata filtering.

The lexical retrieval path applies the equivalent logical filter to BM25 candidates.

This mechanism can later be extended to authenticated enterprise authorization policies.

---

# 13. Grounded Generation

The generation layer receives only retrieved evidence.

The prompt instructs the model to:

- use only provided evidence,
- avoid unsupported outside knowledge,
- cite factual claims,
- avoid inventing provenance,
- abstain when evidence is insufficient.

Flow:

```text
Question
   +
Top-K Evidence
   ↓
Grounded Prompt
   ↓
LLM
   ↓
Answer + Source IDs
```

---

# 14. Backend-Owned Citations

The LLM does not generate trusted filenames or page numbers.

Instead, retrieved evidence is assigned source IDs such as:

```text
[1]
[2]
[3]
```

The model references those IDs.

The backend maps them back to trusted metadata:

```text
[1]
 ↓
source
page
section
chunk_id
```

Example:

```json
{
  "5": {
    "source": "06_trial_protocol_guidance.pdf",
    "page": 7,
    "section": "A. Description and Concept of Master Protocols",
    "chunk_id": "06_trial_protocol_guidance_page_7_chunk_5"
  }
}
```

This avoids relying on the LLM to invent provenance.

---

# 15. Safe Abstention

If the retrieved evidence is insufficient, the system returns:

```text
I could not find sufficient evidence in the provided documents.
```

Example:

```text
What is the capital of Australia?
```

Even if the underlying model knows the answer, the regulatory RAG application should not answer from pretrained knowledge when the approved corpus does not support it.

---

# 16. Observability

Every `/ask` request produces a structured trace.

Captured fields include:

```text
request_id
session_id

original_question
retrieval_query
query_rewritten

retrieval_strategy

rewrite_latency_ms
setup_latency_ms
retrieval_latency_ms
context_build_latency_ms
generation_latency_ms
total_latency_ms

input_tokens
output_tokens
total_tokens

retrieved_chunks

citation_count
citation_valid

abstained
error
```

This enables stage-level debugging.

For a bad answer:

```text
Was correct evidence retrieved?
        │
        ├── No
        │    → Retrieval problem
        │
        └── Yes
             → Generation / prompt problem
```

---

# 17. Cold vs Warm Request Latency

Request-level tracing exposed a significant cold-start cost.

One local cold request showed approximately:

```text
Setup       ~14.1 s
Retrieval    ~1.0 s
Generation   ~3.3 s
Total       ~18.4 s
```

The majority of latency came from retrieval initialization rather than generation.

The routed retriever is therefore cached at application level.

Warm requests generally avoid this initialization cost and typically complete in approximately:

```text
2–5 seconds
```

in the current local environment.

For production, the lexical index would be persisted rather than reconstructed from the PDF corpus at application startup.

---

# 18. Generation Evaluation

Retrieval evaluation answers:

> Did the system find the right evidence?

Generation evaluation answers:

> Did the model use that evidence correctly?

The final generation benchmark contains:

```text
24 questions

18 answerable
6 unanswerable
```

Metrics include:

- expected-term accuracy,
- answer quality,
- faithfulness,
- citation presence,
- citation validity,
- citation completeness,
- citation correctness,
- false abstention rate,
- abstention accuracy.

Run:

```bash
python -m evaluation.evaluate_generation
```

## Final Generation Results

| Metric | Result |
|---|---:|
| Expected-term accuracy | 94.44% |
| Citation presence | 94.44% |
| Citation validity | 100.00% |
| Average answer quality | 2.72 / 3 |
| Average faithfulness | 2.83 / 3 |
| Faithfulness rate | 94.44% |
| Citation completeness | 96.43% |
| Average citation correctness | 2.62 / 3 |
| Fully correct citation rate | 86.39% |
| False abstention rate | 5.56% |
| Abstention accuracy | 100.00% |

---

# 19. Faithfulness Evaluation

Faithfulness measures whether the generated answer is supported by the retrieved evidence.

The evaluator receives:

```text
Original Question
        +
Retrieved Evidence
        +
Generated Answer
        ↓
LLM-as-a-Judge
        ↓
Faithfulness Score
```

The judge is explicitly instructed to use only the retrieved evidence and not outside knowledge.

Scoring:

```text
0 → major unsupported or contradictory claims
1 → several material unsupported claims
2 → mostly supported with minor unsupported detail
3 → every material factual claim is supported
```

The final benchmark achieved:

```text
Average faithfulness score = 2.83 / 3
Faithfulness rate          = 94.44%
```

Faithfulness is currently an **offline evaluation metric** and is not calculated synchronously for every `/ask` request.

This avoids adding an additional LLM judge call to user-facing latency.

---

# 20. Citation Evaluation

Citation quality is evaluated at multiple levels.

## Citation Presence

Checks whether an answer that requires evidence contains citations.

## Citation Validity

Checks whether a generated citation ID actually maps to one of the retrieved evidence blocks.

For example:

```text
Generated:
[3]

Retrieved evidence IDs:
[1] [2] [3] [4] [5]

Result:
VALID
```

But:

```text
Generated:
[7]

Retrieved evidence IDs:
[1] [2] [3] [4] [5]

Result:
INVALID
```

## Citation Completeness

Checks whether material factual claims that require evidence actually contain citations.

## Citation Correctness

Checks whether the specifically cited evidence semantically supports the claim.

This distinction is important:

```text
Citation Validity
        ↓
Does the citation ID exist?

Citation Correctness
        ↓
Does that evidence actually support the claim?
```

A citation can therefore be valid but semantically incorrect.

---

# 21. Retrieval vs Generation Evaluation

The project deliberately evaluates retrieval and generation separately.

```text
                      QUESTION
                          │
                          ▼
                     RETRIEVAL
                          │
               ┌──────────┴──────────┐
               │                     │
               ▼                     │
       Retrieval Evaluation          │
                                     │
       Hit@1 / Hit@3 / Hit@5         │
       MRR                           │
                                     ▼
                                  EVIDENCE
                                     │
                                     ▼
                                    LLM
                                     │
                                     ▼
                                   ANSWER
                                     │
                          ┌──────────┴──────────┐
                          │                     │
                          ▼                     │
                  Generation Evaluation         │
                                                │
                  Answer Quality                │
                  Faithfulness                  │
                  Citations                     │
                  Abstention                    │
```

This separation makes failure analysis easier.

If retrieval metrics regress:

```text
Investigate:
- chunking
- embeddings
- BM25
- filters
- routing
- indexing
```

If retrieval remains stable but generation quality drops:

```text
Investigate:
- generation model
- prompt
- context formatting
- citation instructions
- abstention policy
```

---

# 22. RAG Pipeline Visualizer

The project includes a lightweight browser-based UI designed to expose the internal RAG pipeline rather than hide it behind a traditional chatbot.

The UI displays:

```text
01 · Query Understanding
        ↓
02 · Retrieval Routing
        ↓
03 · Retrieved Evidence
        ↓
04 · Context / Prompt
        ↓
05 · Grounded Answer
        ↓
06 · Trust & Observability
```

## Query Understanding

Displays:

- original question,
- actual retrieval query,
- whether query rewriting occurred.

## Retrieval Routing

Displays:

- Dense or BM25 strategy,
- routing reason,
- document scope,
- retrieval latency.

## Retrieved Evidence

Displays:

- source document,
- page,
- chunk ID,
- retrieval score.

## Grounded Answer

Displays the generated answer together with trusted citation metadata.

## Trust & Observability

Displays:

- citation validation,
- abstention status,
- input/output tokens,
- rewrite latency,
- setup latency,
- retrieval latency,
- context-building latency,
- generation latency,
- total latency.

---

# 23. API

The application is exposed through FastAPI.

## Start the API

```bash
python -m uvicorn src.api.app:app --reload --port 8000
```

The API is available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Example `/ask`

```bash
curl --location 'http://127.0.0.1:8000/ask' \
--header 'Content-Type: application/json' \
--data '{
  "session_id": "demo-1",
  "question": "What is a master protocol?"
}'
```

Expected retrieval strategy:

```text
dense
```

---

# 24. Exact Regulatory Lookup Example

```bash
curl --location 'http://127.0.0.1:8000/ask' \
--header 'Content-Type: application/json' \
--data '{
  "session_id": "demo-cfr",
  "question": "What does 21 CFR 50.25(a)(2) require?"
}'
```

Expected retrieval path:

```text
Exact CFR detected
        ↓
BM25
        ↓
Exact lexical anchor
        ↓
Local context expansion
        ↓
Grounded generation
```

The response trace should show:

```text
retrieval_strategy = bm25
```

---

# 25. Document Filtering Example

```bash
curl --location 'http://127.0.0.1:8000/ask' \
--header 'Content-Type: application/json' \
--data '{
  "session_id": "demo-filter",
  "question": "What risks must be explained during informed consent?",
  "filters": {
    "document_id": "05_informed_consent_guidance"
  }
}'
```

The retrieved evidence should be constrained to the selected document.

---

# 26. Conversational Query Example

First request:

```bash
curl --location 'http://127.0.0.1:8000/ask' \
--header 'Content-Type: application/json' \
--data '{
  "session_id": "conversation-demo",
  "question": "What risks must be explained during informed consent?"
}'
```

Follow-up using the same session:

```bash
curl --location 'http://127.0.0.1:8000/ask' \
--header 'Content-Type: application/json' \
--data '{
  "session_id": "conversation-demo",
  "question": "What about alternatives?"
}'
```

The second request should show:

```text
query_rewritten = true
```

with a standalone retrieval query derived from the conversation history.

---

# 27. Abstention Example

```bash
curl --location 'http://127.0.0.1:8000/ask' \
--header 'Content-Type: application/json' \
--data '{
  "session_id": "abstention-demo",
  "question": "What is the capital of Australia?"
}'
```

Expected behavior:

```text
I could not find sufficient evidence in the provided documents.
```

The trace should show:

```text
abstained = true
```

Since no citations are generated, citation validation is not applicable.

The UI represents this as:

```text
Citation Validation
N/A · NO CITATIONS
```

---

# 28. Running the UI

Keep FastAPI running.

In another terminal:

```bash
cd ui
python -m http.server 5500
```

Open:

```text
http://127.0.0.1:5500
```

The UI communicates with:

```text
http://127.0.0.1:8000/ask
```

If the frontend and API are served on different local ports, ensure FastAPI CORS configuration allows port `5500`.

---

# 29. Local Setup

## Prerequisites

Recommended:

```text
Python 3.11+
```

Create a virtual environment:

```bash
python3.11 -m venv .venv
```

Activate it:

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure the required environment variables, including the OpenAI API key used by the embedding/generation components.

Do not commit secrets to Git.

---

# 30. Verify the Corpus

List the PDFs:

```bash
find data/raw -maxdepth 1 -name "*.pdf" | sort
```

Count them:

```bash
find data/raw -maxdepth 1 -name "*.pdf" | wc -l
```

Expected:

```text
15
```

Current indexed corpus:

```text
Documents : 15
Pages     : 361
Chunks    : 2,534
```

---

# 31. Recommended New Developer Workflow

A new developer can validate the project in the following order.

## Step 1 — Activate Environment

```bash
source .venv/bin/activate
```

## Step 2 — Verify / Build Index

```bash
python -m src.indexing.incremental_indexer
```

Purpose:

```text
Build or incrementally update the knowledge index.
```

## Step 3 — Inspect Retrieval

```bash
python -m src.retrieval.hybrid_retriever
```

Purpose:

```text
Inspect Dense / BM25 / Hybrid retrieval behavior,
scores and returned chunks.
```

## Step 4 — Run Retrieval Evaluation

```bash
python -m evaluation.evaluate_expanded_retrieval
```

Purpose:

```text
Validate:
Hit@1
Hit@3
Hit@5
MRR

across:
Dense
BM25
Hybrid
Routed
```

## Step 5 — Run Generation Evaluation

```bash
python -m evaluation.evaluate_generation
```

Purpose:

```text
Validate:
answer quality
faithfulness
citations
abstention
```

This step makes multiple LLM calls and is therefore slower than retrieval-only evaluation.

## Step 6 — Start API

```bash
python -m uvicorn src.api.app:app --reload --port 8000
```

## Step 7 — Start UI

In another terminal:

```bash
cd ui
python -m http.server 5500
```

Then open:

```text
http://127.0.0.1:5500
```

---

# 32. Command Reference

| Command | Purpose |
|---|---|
| `python -m src.indexing.incremental_indexer` | Build/update document index |
| `python -m src.retrieval.hybrid_retriever` | Inspect/debug retrieval |
| `python -m evaluation.evaluate_retrieval` | Initial chunking/retrieval experiments |
| `python -m evaluation.evaluate_expanded_retrieval` | Final retrieval benchmark |
| `python -m evaluation.evaluate_generation` | End-to-end generation evaluation |
| `python -m uvicorn src.api.app:app --reload --port 8000` | Start API |
| `python -m http.server 5500` | Serve UI from `ui/` |

---

# 33. Key Engineering Findings

This project evolved through measured failure analysis.

## Finding 1 — Chunk Size Matters

Chunk size was evaluated experimentally.

The initial benchmark favored:

```text
500 / 50
```

rather than selecting a chunk size based on convention.

## Finding 2 — Dense Is Not Enough

Dense retrieval performed well for conceptual questions but struggled with exact regulatory identifiers.

## Finding 3 — BM25 Is Not Enough

BM25 solved exact identifier lookup but could retrieve only a regulatory footnote without enough surrounding explanatory context.

## Finding 4 — Hybrid Is Not Automatically Better

RRF improved some recall but did not outperform query-aware routing overall.

## Finding 5 — Retrieval Should Be Query-Aware

The final strategy uses:

```text
Semantic → Dense

Exact CFR → BM25 + local context expansion
```

## Finding 6 — Citation Validity Is Not Citation Correctness

A citation can point to a real retrieved chunk but still fail to fully support the associated claim.

Both need separate evaluation.

## Finding 7 — Abstention Is a Feature

For a grounded regulatory assistant, refusing to answer without evidence is safer than relying on pretrained model knowledge.

## Finding 8 — Observability Changes Architecture

Stage-level tracing identified cold-start initialization—not the LLM—as the dominant latency source on the first request.

---

# 34. Production Considerations

The current project is designed as a local engineering prototype with production-oriented patterns.

For a larger enterprise deployment, the following changes would be appropriate.

## Vector Infrastructure

Move from local Qdrant storage to a managed/server-based Qdrant deployment.

## Lexical Retrieval

Move in-memory BM25 to a persisted search engine such as:

```text
OpenSearch
or
Elasticsearch
```

## Session State

Move application-memory conversation state to:

```text
Redis
or
a durable database
```

## Authentication & Authorization

Introduce:

```text
User Authentication
        ↓
Authorization Policy
        ↓
Allowed Document Scope
        ↓
Metadata Filter
        ↓
Retrieval
```

Authorization should be applied before evidence reaches the LLM.

## Ingestion

Move document processing to asynchronous workers:

```text
Document Upload
      ↓
Object Storage
      ↓
Queue
      ↓
Indexing Worker
      ↓
Parse / Chunk / Embed
      ↓
Vector + Lexical Indexes
```

## Versioning

Version:

- document corpus,
- embedding model,
- retrieval configuration,
- prompts,
- evaluation datasets.

This enables regression detection and rollback.

---

# 35. Potential Future Improvements

Potential extensions include:

- persisted BM25/OpenSearch indexing,
- cross-encoder reranking,
- token-aware chunking,
- semantic chunking experiments,
- multi-evidence Recall@K evaluation,
- nDCG evaluation,
- query-router expansion,
- authorization-aware retrieval,
- Redis-backed sessions,
- background ingestion workers,
- distributed tracing,
- prompt/index versioning,
- evaluation gates in CI/CD,
- sampled production faithfulness evaluation,
- retrieval caching,
- model routing,
- deployment on Kubernetes/EKS.

---

# 36. Evaluation Philosophy

The project intentionally avoids optimizing every benchmark metric to 100%.

The goal of evaluation is to expose failure modes.

The development loop is:

```text
Build
  ↓
Measure
  ↓
Inspect Failures
  ↓
Identify Root Cause
  ↓
Change Architecture
  ↓
Evaluate Again
```

Examples from this project:

```text
Dense exact-CFR failure
        ↓
BM25
```

```text
BM25 footnote-only evidence
        ↓
Context expansion
```

```text
Hybrid not best overall
        ↓
Query-aware routing
```

```text
Cold request latency
        ↓
Stage-level tracing
        ↓
Retriever caching
```

---

# 37. Core Takeaway

The project demonstrates that production-oriented RAG is not simply:

```text
Embedding + Vector DB + LLM
```

A more complete system requires:

```text
Document Lifecycle
        +
Chunking Strategy
        +
Semantic Retrieval
        +
Lexical Retrieval
        +
Query Routing
        +
Conversation Handling
        +
Grounded Generation
        +
Provenance
        +
Safe Abstention
        +
Observability
        +
Evaluation
```

The most important architectural decisions in this project were driven by **measured retrieval and generation failures rather than assumptions**.

---

# Demo Scenarios

### Semantic Retrieval

> What is a master protocol?

Expected route: `Dense`

### Exact Regulatory Retrieval

> What does 21 CFR 50.25(a)(2) require?

Expected route: `BM25 + Context Expansion`

### Conversational Follow-Up

> What risks must be explained during informed consent?

followed by:

> What about alternatives?

Expected: `query_rewritten = true`

### Safe Abstention

> What is the capital of Australia?

Expected: grounded abstention.

# Disclaimer

This repository is an educational/engineering demonstration of RAG architecture and evaluation techniques.

The included regulatory and clinical guidance documents are used as a knowledge corpus for experimentation.

Generated answers should not be treated as medical, clinical, legal, or regulatory advice.