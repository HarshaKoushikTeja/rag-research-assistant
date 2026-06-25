import os
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from chunk import get_all_chunks

# Load DB credentials from .env
load_dotenv()

def get_connection():
    return psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )

# 1. Load chunks and embedding model
print("Loading chunks...")
chunks = get_all_chunks()
print(f"  {len(chunks)} chunks loaded.")

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding all chunks...")
texts = [c["text"] for c in chunks]
vectors = model.encode(texts, show_progress_bar=True)
print(f"  Done. Shape: {vectors.shape}")

# 2. Connect and store
print("Connecting to database...")
with get_connection() as conn:
    register_vector(conn)  # teach psycopg about the vector type
    with conn.cursor() as cur:
        # Clear any existing rows so re-running gives a clean state
        cur.execute("TRUNCATE TABLE chunks RESTART IDENTITY;")

        # Insert every chunk + its vector
        for chunk, vector in zip(chunks, vectors):
            cur.execute(
                "INSERT INTO chunks (paper, chunk_index, content, embedding) VALUES (%s, %s, %s, %s)",
                (chunk["paper"], chunk["index"], chunk["text"], vector),
            )
    conn.commit()  # save all the inserts

print(f"Stored {len(chunks)} chunks in the database.")