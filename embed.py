from fastembed import TextEmbedding
import numpy as np

model = TextEmbedding("BAAI/bge-small-en-v1.5")

def embed(texts):
    vecs = np.array(list(model.embed(texts)), dtype=np.float32)
    # normalize each vector so cosine similarity = dot product later
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)

if __name__ == "__main__":
    test = ["The laws of human nature", "Power comes from self-mastery"]
    vecs = embed(test)
    print(f"Shape: {vecs.shape}")        # should be (2, 384)
    print(f"First vector (first 5 values): {vecs[0][:5]}")
    # dot product of a vector with itself should be ~1.0 (normalized)
    print(f"Self-similarity: {np.dot(vecs[0], vecs[0]):.4f}")