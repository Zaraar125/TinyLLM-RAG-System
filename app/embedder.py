from sentence_transformers import SentenceTransformer

# Using a small embedding model (you can swap it later)
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    return model.encode(texts, show_progress_bar=True)
