from sentence_transformers import SentenceTransformer
import numpy as np

model=SentenceTransformer("all-MiniLM-L6-v2")
sentences=["I'm applying for AI engineering internships for Spring 2027.",
    "I'm looking for a machine learning internship next year.",
    "The recipe calls for two cups of flour and a teaspoon of salt.",]

embeddings= model.encode(sentences)
print(" shape of embeddings:", embeddings.shape)
print("First 5 numbers of embedding 1:",  embeddings[0][:5])


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


sim_internship_pair = cosine_similarity(embeddings[0], embeddings[1])
sim_internship_recipe = cosine_similarity(embeddings[0], embeddings[2])

print(f"\nSimilarity (AI internship vs ML internship): {sim_internship_pair:.4f}")
print(f"Similarity (AI internship vs recipe): {sim_internship_recipe:.4f}")