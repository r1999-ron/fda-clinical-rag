from src.embeddings.embedder import get_embedding_model


embedding_model = get_embedding_model()

vector = embedding_model.embed_query(
    "What is the adverse event reporting policy?"
)

print("Vector type:", type(vector))
print("Dimensions:", len(vector))
print("First 10 values:")
print(vector[:10])