import faiss
import numpy as np
import pickle

from ingest import ingest_documents

records, embeddings = ingest_documents()

embeddings = embeddings.astype("float32")

faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])

index.add(embeddings)

faiss.write_index(index, "faiss_index.bin")
with open("records.pkl", "wb") as f:
    pickle.dump(records, f)

print(f"Indexed {index.ntotal} vectors")