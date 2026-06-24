# RAG Research Assistant

An end-to-end Retrieval-Augmented Generation (RAG) system that answers questions about foundational AI research papers using semantic search and source-grounded retrieval.

Built with a production-oriented mindset, this project focuses on the core components behind modern AI knowledge systems: document ingestion, chunking, embeddings, vector retrieval, evaluation, and evidence-based iteration.

> **Status:** In active development, built phase by phase. The retrieval foundation is working; answer generation, persistence, and evaluation are in progress. See the roadmap below.

---

## Project Highlights

- Ingested and indexed **4 foundational AI research papers** from arXiv
- Generated **497 searchable document chunks** with overlap for context preservation
- Created **384-dimensional semantic embeddings** using Sentence Transformers
- Built an end-to-end semantic retrieval pipeline (query -> embedding -> cosine similarity -> ranked passages)
- Implemented chunk-level metadata tracking for source attribution
- **Identified and documented real retrieval failure modes** (see [\`sample_output.md\`](sample_output.md)) to drive measurable improvements
- Designed with a production roadmap including evaluation, reranking, pgvector, FastAPI, and Docker

---

## Why This Project?

Modern LLM applications increasingly rely on Retrieval-Augmented Generation (RAG) to reduce hallucinations and provide grounded, citable responses.

Rather than building a one-off notebook demo, this project focuses on understanding and **iteratively improving every layer** of a RAG pipeline -- and measuring whether each improvement actually helps, rather than assuming it does.

---

## Example Query

> **Question:** What is multi-head attention?

The system embeds the query, runs cosine similarity against all chunk embeddings, and returns the top-ranked passages with source metadata. The actual top result for this query came from *Attention Is All You Need*, and the real definition of multi-head attention was retrieved at rank 2.

**The full, unedited retrieval output -- including its imperfections and what they reveal -- is documented in [\`sample_output.md\`](sample_output.md).** Those imperfections are deliberately surfaced rather than hidden, because they define the measurable targets for the planned reranking and structure-aware chunking improvements.

---

## System Architecture

\`\`\`text
+--------------------+
| AI Research Papers |
|      (arXiv)       |
+---------+----------+
          v
+--------------------+
| Document Ingestion |  requests + BeautifulSoup
+---------+----------+
          v
+--------------------+
| Text Cleaning      |  lossless reformatting only
+---------+----------+
          v
+--------------------+
| Chunking Pipeline  |  fixed-size + overlap, with metadata
+---------+----------+
          v
+--------------------+
| Embedding Model    |  all-MiniLM-L6-v2 (384-dim)
+---------+----------+
          v
+--------------------+
| Vector Retrieval   |  cosine similarity, top-K
+---------+----------+
          v
+--------------------+
| Grounded Answers   |  (planned)
| + Citations        |
+--------------------+
\`\`\`

---

## Corpus

| Paper                                                            | arXiv ID   |
| ---------------------------------------------------------------- | ---------- |
| Attention Is All You Need                                        | 1706.03762 |
| BERT: Pre-training of Deep Bidirectional Transformers            | 1810.04805 |
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | 2005.11401 |
| Language Models are Few-Shot Learners (GPT-3)                    | 2005.14165 |

Paper texts are not committed to the repository. Run \`python ingest.py\` to fetch and build the corpus from the source URLs.

---

## Tech Stack

**AI / NLP:** Sentence Transformers, transformer embeddings, semantic search, dense vector retrieval

**Backend:** Python 3.11, requests, BeautifulSoup, scikit-learn

**Infrastructure (planned):** PostgreSQL + pgvector, FastAPI, Docker

---

## Pipeline Components

### 1. Document Ingestion -- \`ingest.py\`
- Downloads paper content from ar5iv HTML renderings
- Extracts article text with BeautifulSoup (drops scripts, styles, nav, footer)
- Applies **lossless** cleaning: tidies citation brackets split across lines, collapses redundant blank lines
- Preserves research content; does not strip math or reference noise (see *Design Decisions*)

### 2. Chunking -- \`chunk.py\`
- Splits papers into ~1000-character chunks with ~150-character overlap
- Stores per-chunk metadata (\`paper\`, \`index\`) for source attribution
- Importable module (\`get_all_chunks()\`) reused by downstream scripts
- Current corpus: **4 papers, 497 chunks**

### 3. Embedding + Retrieval -- \`embed.py\`
- Embeds all chunks and the query with \`all-MiniLM-L6-v2\` into a shared 384-dim space
- Ranks chunks by cosine similarity and returns the top-K with source metadata

---

## Design Decisions

**Local embeddings over an API.** \`all-MiniLM-L6-v2\` runs locally at zero cost, enabling unlimited free re-runs during tuning. It is treated as a defensible *baseline*, not a final choice -- the embedding model is a planned, measurable upgrade once evaluation exists.

**Fixed-size chunking first.** A simple, robust baseline before introducing structure-aware or semantic chunking. Each future strategy will be measured against evaluation metrics rather than assumed beneficial.

**Lossless cleaning only.** Only transformations that provably preserve information are applied (citation formatting, blank-line normalization). Aggressive cleaning such as stripping mathematical notation was deliberately rejected: superscripts mark both footnotes *and* real exponents, so a blanket strip would destroy content. Cosmetic noise that does not affect retrieval was intentionally left in.

---

## Observations and Findings

Testing the query *"What is multi-head attention?"* against the baseline surfaced two real retrieval issues (full output in [\`sample_output.md\`](sample_output.md)):

- **Ranking errors** -- the exact definition did not always rank first. *Planned improvement:* cross-encoder reranking.
- **Bibliography false positives** -- reference sections occasionally scored highly due to overlapping academic vocabulary. *Planned improvement:* structure-aware chunking / section filtering.

These findings provide concrete, measurable targets for the evaluation and improvement phases.

---

## AI Engineering Concepts Demonstrated

| Area                  | Concepts                    |
| --------------------- | --------------------------- |
| NLP                   | Transformer embeddings      |
| RAG                   | Dense retrieval             |
| Information Retrieval | Vector search               |
| Machine Learning      | Embedding models            |
| Data Engineering      | Ingestion pipelines         |
| Evaluation            | Retrieval quality analysis  |
| MLOps                 | Experiment-driven iteration |

---

## Project Roadmap

**Phase 1 -- Retrieval Foundation** (complete)
Corpus selection - ingestion pipeline - chunking - embedding generation - semantic retrieval

**Phase 2 -- Evaluation Framework** (in progress)
Faithfulness - context precision - retrieval recall - answer relevance

**Phase 3 -- Retrieval Improvements**
Structure-aware chunking - hybrid retrieval (BM25 + dense) - cross-encoder reranking -- each validated against the evaluation harness

**Phase 4 -- End-to-End RAG**
LLM answer generation - inline source citations - out-of-scope ("I don't know") detection

**Phase 5 -- Production Deployment**
PostgreSQL + pgvector (persistent storage, dynamic document addition) - FastAPI service - Docker - observability

---

## Repository Structure

\`\`\`text
.
|- ingest.py          # fetch, extract, clean papers -> data/
|- chunk.py           # chunk cleaned text (importable: get_all_chunks)
|- embed.py           # embed chunks + cosine-similarity retrieval
|- requirements.txt
|- sample_output.md   # real retrieval output + analysis
|- README.md
\`- data/              # generated by ingest.py (not committed)
\`\`\`

---

## Setup

\`\`\`bash
# Clone and enter
git clone https://github.com/HarshaKoushikTeja/rag-research-assistant.git
cd rag-research-assistant

# Virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / Mac

# Dependencies
pip install -r requirements.txt

# Build the corpus, then run retrieval
python ingest.py               # fetches + cleans the 4 papers into data/
python embed.py                # embeds chunks (imports chunking internally) and runs a sample query
\`\`\`

---

## Future Vision

The long-term goal is to evolve this from a retrieval prototype into a production-grade AI knowledge system supporting grounded question answering, multi-document reasoning, source attribution, hybrid search, and agentic workflows -- built by focusing on **measurable improvements rather than feature accumulation**.

---

## Author

**Harsha Koushik Teja Aila**
MS in Data Science, Analytics & Engineering -- Arizona State University

Interested in AI Engineering, Machine Learning, RAG, LLM systems, MLOps, and Software Engineering.

- GitHub: [HarshaKoushikTeja](https://github.com/HarshaKoushikTeja)
- LinkedIn: [aila-harsha-koushik-teja](https://www.linkedin.com/in/aila-harsha-koushik-teja)
- Portfolio: [harshaaila.netlify.app](https://harshaaila.netlify.app)
