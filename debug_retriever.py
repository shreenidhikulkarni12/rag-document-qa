
import sys
from retriever import retrieve

question = " ".join(sys.argv[1:]) or "What is activity theory?"

for r in retrieve(question, top_k=5):
    print(f"[{r['score']:.4f}] {r['source']} (chunk {r['chunk_id']})")
    print(r["text"][:250])
    print("-" * 60)