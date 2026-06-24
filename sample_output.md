# Sample Output

This file shows **real, unedited output** from the current retrieval baseline — not idealized results. The imperfections shown here are intentional: they document how a fixed-size + bi-encoder baseline actually behaves, and they motivate the planned improvements (reranking, structure-aware chunking), which will be validated with before/after metrics in later phases.

---

## Query

> What is multi-head attention?

## Configuration

| Setting | Value |
| --- | --- |
| Embedding model | `all-MiniLM-L6-v2` (384-dim) |
| Chunking | ~1000 chars, ~150 char overlap |
| Retrieval | cosine similarity, top-3 |
| Corpus | 497 chunks across 4 papers |

---

## Retrieved Passages (top 3)

**Rank 1 — `attention`, chunk 51 — similarity 0.611**

> "...word 'its' for attention heads 5 and 6. Note that the attentions are very sharp for this word. Figure 5: Many of the attention heads exhibit behaviour that seems related to the structure of the sentence. We give two such examples above, from two different heads from the encoder self-attention at layer 5 of 6. The heads clearly learned to perform different tasks. ..."

**Rank 2 — `attention`, chunk 14 — similarity 0.589**

> "...2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel. Instead of performing a single attention function with d model [math-formatting noise] -dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections ..."

**Rank 3 — `attention`, chunk 50 — similarity 0.586**

> "...ne translation. CoRR, abs/1606.04199, 2016. [40] Muhua Zhu, Yue Zhang, Wenliang Chen, Min Zhang, and Jingbo Zhu. Fast and accurate shift-reduce constituent parsing. In Proceedings of the 51st Annual Meeting of the ACL ... Attention Visualizations Figure 3: An example of the attention mechanism following long-distance dependencies in the encod ..."

---

## Analysis

### What worked

All three top results came from the **correct paper** (*Attention Is All You Need*) out of four in the corpus — retrieval correctly identified the relevant source by *meaning*, with no keyword matching. Rank 2 contains the actual definition of multi-head attention.

### What did not

Three real failure modes appear:

**1. The best answer ranked second, not first.** The definition (Rank 2, 0.589) was outscored by a chunk about attention-head *visualizations* (Rank 1, 0.611). A bi-encoder's single similarity score is a coarse ranker.
- *Planned fix:* a cross-encoder **reranker** as a second-stage scorer (Phase 3), measured against the evaluation harness (Phase 2).

**2. A reference list scored into the top 3.** Rank 3 is largely bibliography text, which scored highly because citation vocabulary superficially resembles academic-question vocabulary.
- *Planned fix:* **structure-aware chunking** to detect and down-weight non-content sections such as References (Phase 3).

**3. Math-formatting noise in the definition chunk.** The retrieved definition contains artifacts from the source HTML's math rendering (shown above as "[math-formatting noise]" — in the raw output it appears as duplicated LaTeX/Unicode for the symbol *d_model*). This was a deliberate ingestion decision: removing it cleanly is non-trivial (superscripts mark both footnotes and real exponents), and it does not prevent the chunk from being retrieved. See *Design Decisions* in the README.

---

These observations are the baseline measurements that the planned improvements will be evaluated against. When reranking and structure-aware chunking are added, this same query will be re-run and the before/after numbers recorded here.
