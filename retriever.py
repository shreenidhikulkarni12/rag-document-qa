import faiss
import pickle

from src.embedder import embed_query

index = faiss.read_index("faiss_index.bin")
with open("records.pkl", "rb") as f:
    records = pickle.load(f)

# "HCC_3_Shrinidhi_Kulkarni.pdf" -> "hcc_3"
SOURCE_KEYS = {
    r["source"]: "_".join(r["source"].split("_")[:2]).lower()
    for r in records
}


def find_source(question: str):
    q = question.lower().replace(" ", "_")
    for source, key in SOURCE_KEYS.items():
        if key in q:
            return source
    return None


def retrieve(question: str, top_k: int = 3):
    source = find_source(question)

    query_vector = embed_query(question).astype("float32").reshape(1, -1)
    faiss.normalize_L2(query_vector)

    # If a document is named, search everything and filter afterwards
    k = index.ntotal if source else top_k
    scores, indices = index.search(query_vector, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        record = records[idx]
        if source and record["source"] != source:
            continue
        results.append({
            "text": record["text"],
            "source": record["source"],
            "chunk_id": record["chunk_id"],
            "score": float(score),
        })
        if len(results) == top_k:
            break
    return results