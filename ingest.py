from pathlib import Path

from src.loader import load_text
from src.chunker import chunk_text
from src.embedder import embed_chunks

DOCS_DIR = Path("documents")


def ingest_documents():
    records = []

    for path in sorted(DOCS_DIR.iterdir()):
        if path.suffix.lower() not in (".pdf", ".txt"):
            continue

        text = load_text(path.name, path.read_bytes())
        text = " ".join(text.split()) 
        if not text.strip():
            print(f"{path.name}: no extractable text, skipping")
            continue

        chunks = chunk_text(text, max_tokens=200, overlap_tokens=30)

        for i, chunk in enumerate(chunks):
            records.append({"text": chunk, "source": path.name, "chunk_id": i})

        print(f"{path.name}: {len(chunks)} chunks")

    embeddings = embed_chunks([r["text"] for r in records])
    return records, embeddings


if __name__ == "__main__":
    records, embeddings = ingest_documents()
    print("Total chunks:", len(records))
    print("Embeddings shape:", embeddings.shape)
    print("Sample record:", records[0])