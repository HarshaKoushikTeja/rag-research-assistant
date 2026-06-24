# RAG Research Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions about foundational AI research papers, grounding every answer in the source text. Built incrementally, phase by phase, with an emphasis on production thinking — evaluation, source citations, and evidence-driven iteration rather than a one-off notebook demo.

> **Status:** In active development. Phase 1 (core retrieval pipeline) partially complete — see the progress table below.

## What it does

Ask a natural-language question (e.g. *"What is multi-head attention?"*) and the system retrieves the most semantically relevant passages from a corpus of AI papers and (in later phases) generates a cited answer grounded in those passages.

## Corpus

Four foundational NLP/LLM papers, ingested from arXiv's clean HTML (ar5iv) renderings:

| Paper | arXiv ID |
|-------|----------|
| Attention Is All You Need | 1706.03762 |
| BERT | 1810.04805 |
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | 2005.11401 |
| Language Models are Few-Shot Learners (GPT-3) | 2005.14165 |

## Tech stack

- **Python 3.11**
- **requests + BeautifulSoup** — document ingestion (HTML fetch and text extraction)
- **sentence-transformers (`all-MiniLM-L6-v2`)** — embeddings (384-dim)
- **scikit-learn** — cosine similarity for retrieval
- **pgvector / PostgreSQL** *(upcoming)* — persistent vector storage
- **FastAPI** *(upcoming)* — serving layer
- **Docker** *(upcoming)* — deployment

## Architecture (current)

```
arXiv HTML  ->  extract + clean  ->  chunk (fixed-size + overlap)  ->  embed  ->  cosine similarity retrieval
```

## Pipeline details

**Ingestion** (`ingest.py`) — Fetches each paper's ar5iv HTML, extracts the article text with BeautifulSoup (dropping script/style/nav/footer), and applies two deliberately **lossless** cleaning rules: tidying citation brackets that were split across lines, and collapsing redundant blank lines. Math notation and reference noise are intentionally left untouched (see *Design decisions* below).

**Chunking** (`chunk.py`) — Splits each paper into ~1000-character chunks with ~150-character overlap. Each chunk carries metadata (`paper`, `index`) to support source citations downstream. Current corpus: **497 chunks** across four papers.

**Embedding + retrieval** (`embed.py`) — Embeds all chunks with MiniLM, embeds the query, and ranks chunks by cosine similarity.

## Progress

| Phase | Step | Status |
|-------|------|--------|
| 1 | Corpus selection + environment | Done |
| 1 | Ingestion (extract + clean) | Done |
| 1 | Chunking (fixed-size + overlap) | Done |
| 1 | Embedding (MiniLM, 384-dim) | Done |
| 1 | In-memory semantic retrieval | Done |
| 1 | Persistent vector store (pgvector) | Planned |
| 1 | LLM answer generation + citations | Planned |
| 2 | Evaluation harness (faithfulness, relevance, context precision) | Planned |
| 3 | Structure-aware chunking (measured upgrade) | Planned |
| 3 | Hybrid retrieval (BM25 + dense) + reranking | Planned |
| 4 | Agent layer (tool use, memory) | Planned |
| 4+ | Multi-source retrieval, observability, deployment | Planned |

## Design decisions

- **Local embeddings over an API.** `all-MiniLM-L6-v2` runs locally at zero cost, enabling free re-runs during tuning. Chosen as a defensible baseline, not a final answer — the embedding model is a planned, measurable upgrade once evaluation exists.
- **Fixed-size chunking first.** A simple, robust baseline. Structure-aware chunking is planned as a *measured* improvement (compare retrieval quality before/after), not assumed-better upfront.
- **Minimal, lossless cleaning.** Only reformatting that provably removes no information. Aggressive cleaning (e.g. stripping math) was rejected: superscripts mark both footnotes and exponents, so a blanket strip would destroy content. Cosmetic noise that does not affect retrieval was deliberately left in.

## Known issues / observations

Observed while testing the query *"What is multi-head attention?"* against the baseline:

- **The best answer did not rank first.** The literal definition of multi-head attention ranked #2, behind a tangentially related chunk. *Likely fix:* a cross-encoder **reranker** as a second-stage scorer (planned, Phase 3).
- **Reference-list text surfaces as a false positive.** A chunk consisting largely of a bibliography scored into the top results because citation vocabulary superficially resembles academic-question vocabulary. *Likely fix:* **structure-aware chunking** to identify and down-weight non-content sections like References (planned, Phase 3).

These are expected behaviours of a fixed-size + bi-encoder baseline and are being used to motivate the planned upgrades with measured before/after evidence.

## Future improvements

- Persistent vector storage (pgvector) so embeddings are computed once, not per run — also the foundation for dynamic document addition
- LLM-generated answers with inline source citations and "I don't know" handling for out-of-corpus questions
- Retrieval evaluation harness (faithfulness, answer relevance, context precision)
- Structure-aware chunking, hybrid retrieval, and reranking — each validated against the evaluation harness
- Agentic layer with tool use and conversational memory
- Multi-source retrieval (combining vector search with structured/SQL sources)
- Observability and containerized deployment

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (use: source venv/bin/activate on Mac/Linux)
pip install requests beautifulsoup4 sentence-transformers scikit-learn

python ingest.py               # fetch + clean the four papers into data/
python embed.py                # embed chunks and run a sample retrieval query
```

*Paper texts are not committed — run `ingest.py` to populate `data/` from the source URLs.*
