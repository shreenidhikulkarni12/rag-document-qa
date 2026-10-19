from sentence_transformers import SentenceTransformer
model= SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks( chunks:list[str]):
     return model.encode(chunks)


def embed_query(question: str):
    return model.encode(question)


