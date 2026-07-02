import re
from rank_bm25 import BM25Okapi
from chunk import get_all_chunks

def tokenize(text):
    """Lowercase and split into word tokens, splitting on any non-alphanumeric character."""
    return re.findall(r"[a-z0-9]+", text.lower())

# Load chunks once and build the BM25 index
print("Loading chunks and building BM25 index...")
chunks = get_all_chunks()
tokenized_corpus = [tokenize(c["text"]) for c in chunks]
bm25 = BM25Okapi(tokenized_corpus)
print(f"  BM25 index built over {len(chunks)} chunks.")


def bm25_retrieve(question, top_k=3):
    """Return the top_k chunks ranked by BM25 keyword score."""
    tokenized_query = tokenize(question)
    scores = bm25.get_scores(tokenized_query)

    # pair each chunk with its score, sort descending, take top_k
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for chunk, score in ranked:
        results.append((chunk["paper"], chunk["index"], chunk["text"], score))
    return results


if __name__ == "__main__":
    # Test on questions we KNOW dense retrieval struggled with
    test_questions = [
        "What optimizer was used to train the Transformer?",
        "What learning rate warmup schedule does the Transformer use?",
        "What does BERT stand for?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {q}")
        results = bm25_retrieve(q, top_k=3)
        for rank, (paper, idx, content, score) in enumerate(results, 1):
            print(f"  Rank {rank} | {paper} chunk {idx} | BM25 score {score:.2f}")
            print(f"    {content[:150]}")