import os
import glob

CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 150    # characters repeated from the end of the previous chunk


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping fixed-size chunks."""
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # advance the window, but step BACK by `overlap` so the next
        # chunk repeats the tail of this one
        start = end - overlap

    return chunks


# --- Process every cleaned paper in data/ ---
def get_all_chunks(data_dir="data"):
    """Read every cleaned .txt, chunk it, return a list of chunk dicts."""
    all_chunks = []
    for filepath in sorted(glob.glob(f"{data_dir}/*.txt")):
        name = os.path.basename(filepath).replace(".txt", "")
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({"paper": name, "index": i, "text": chunk})

    return all_chunks


# This block runs ONLY when you do `python chunk.py` directly,
# NOT when another script does `import chunk`.
if __name__ == "__main__":
    chunks = get_all_chunks()

    # per-paper counts
    from collections import Counter
    counts = Counter(c["paper"] for c in chunks)
    for paper, n in counts.items():
        print(f"{paper}: {n} chunks")
    print(f"\nTotal chunks across all papers: {len(chunks)}")

    print("\n--- First chunk preview ---")
    print(chunks[0]["text"][:300])
    print("\n--- Second chunk preview (note the overlap) ---")
    print(chunks[1]["text"][:300])