import openai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# OpenAI API Key
openai.api_key = "5PGUZoFqwJK96jASVRMAeb8HJtoMUvlvMSvA8eTniDqvLkdyiBhnJQQJ99ALACYeBjFXJ3w3AAABACOGYDFn"

# Function to generate ADA embeddings
def get_ada_embeddings(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

# Example sentences
sentence1 = "I love using AI tools."
sentence2 = "AI tools are helpful for productivity."

# Generate embeddings
embedding1 = get_ada_embeddings(sentence1)
embedding2 = get_ada_embeddings(sentence2)

# Compute cosine similarity
similarity = cosine_similarity([embedding1], [embedding2])
print(f"Cosine Similarity: {similarity[0][0]:.2f}")
