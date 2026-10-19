import ollama

from retriever import retrieve

MODEL = "llama3.2"
MIN_SCORE = 0.35  # chunks scoring below this are treated as irrelevant


def answer_question(question: str, top_k: int = 5):
    results = [r for r in retrieve(question, top_k=top_k) if r["score"] >= MIN_SCORE]

    if not results:
        return {"answer": "I don't know.", "sources": []}

    context = "\n\n".join(
        f"[Source: {r['source']}]\n{r['text']}" for r in results
    )

    messages = [
        {
            "role": "system",
            "content":"Answer the question using ONLY the context provided below. "
           "If the answer isn't in the context, say you don't know. "
           "Do not use outside knowledge.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]

    response = ollama.chat(model=MODEL, messages=messages)
    answer = response["message"]["content"]

    sources = list({r["source"] for r in results})

    if answer.strip().lower().startswith("i don't know"):
        sources = []

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    result = answer_question("What is activity theory?")
    print("ANSWER:")
    print(result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"- {s}")