import os
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )

# Load the model once (needed only to embed the question, not the corpus)
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(question, top_k=3):
    """Embed the question and fetch the top_k most similar chunks from pgvector."""
    q_vector = model.encode(question)

    with get_connection() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT paper, chunk_index, content,
                       1 - (embedding <=> %s) AS similarity
                FROM chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (q_vector, q_vector, top_k),
            )
            return cur.fetchall()


if __name__ == "__main__":
    question = "What is multi-head attention?"
    results = retrieve(question)

    print(f"Question: {question}\n")
    for rank, (paper, idx, content, similarity) in enumerate(results, 1):
        print(f"--- Rank {rank} | {paper} chunk {idx} | similarity {similarity:.3f} ---")
        print(content[:400])
        print()