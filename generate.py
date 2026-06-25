import ollama
from retrieve import retrieve

MODEL = "gemma3:4b"


def build_prompt(question, chunks):
    """Construct a grounded RAG prompt from retrieved chunks."""
    # Number each chunk and label its source, so the model can cite
    context_blocks = []
    for i, (paper, idx, content, similarity) in enumerate(chunks, 1):
        context_blocks.append(f"[Source {i}: {paper}, chunk {idx}]\n{content}")
    context = "\n\n".join(context_blocks)

    prompt = f"""You are a research assistant answering questions about AI papers. Use ONLY the context below to answer the question. If the context does not contain the answer, say "I don't have enough information in the provided context to answer that." Do not use outside knowledge.

When you use information from a source, cite it like [Source 1].

Context:
{context}

Question: {question}

Answer:"""
    return prompt


def answer(question, top_k=3):
    chunks = retrieve(question, top_k=top_k)
    prompt = build_prompt(question, chunks)

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    answer_text = response["message"]["content"]

    return answer_text, chunks


if __name__ == "__main__":
    question = "What is multi-head attention?"
    answer_text, chunks = answer(question)

    print(f"Question: {question}\n")
    print("Answer:")
    print(answer_text)
    print("\n" + "=" * 60)
    print("Sources used:")
    for i, (paper, idx, content, similarity) in enumerate(chunks, 1):
        print(f"  [Source {i}] {paper}, chunk {idx} (similarity {similarity:.3f})")