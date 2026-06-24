from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from chunk import get_all_chunks

# 1. Load chunks and model
print("Loading chunks...")
chunks = get_all_chunks()
print(f"  {len(chunks)} chunks loaded.")

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Embed every chunk's text (this is the indexing step)
print("Embedding all chunks (takes a few seconds)...")
chunk_texts = [c["text"] for c in chunks]
chunk_vectors = model.encode(chunk_texts, show_progress_bar=True)
print(f"  Done. Shape: {chunk_vectors.shape}")  # (497, 384)

# 3. Ask a question -> embed it -> find closest chunks
question = "What is multi-head attention?"
q_vector = model.encode([question])  # note: list, so shape is (1, 384)

# cosine similarity between the question and ALL chunk vectors at once
similarities = cosine_similarity(q_vector, chunk_vectors)[0]  # array of 497 scores

# get the indices of the top 3 most similar chunks
top_k = 3
top_indices = np.argsort(similarities)[::-1][:top_k]

# 4. Show the results
print(f"\nQuestion: {question}\n")
print(f"Top {top_k} most relevant chunks:\n")
for rank, idx in enumerate(top_indices, 1):
    c = chunks[idx]
    score = similarities[idx]
    print(f"--- Rank {rank} | {c['paper']} chunk {c['index']} | similarity {score:.3f} ---")
    print(c["text"][:400])
    print()