# import numpy as np
# from sentence_transformers import SentenceTransformer
# import faiss
# import pickle

# # Load model once
# model = SentenceTransformer("all-MiniLM-L6-v2")

# def embed_query(query):
#     return model.encode([query])[0]  # Single vector

# def retrieve_top_k(query, k=3, vector_store_path="vector_store"):
#     # Load FAISS index
#     index = faiss.read_index(f"{vector_store_path}.index")

#     # Load corresponding chunks
#     with open(f"{vector_store_path}_texts.pkl", "rb") as f:
#         texts = pickle.load(f)

#     # Embed the query
#     query_vec = embed_query(query).astype("float32").reshape(1, -1)

#     # Perform similarity search
#     D, I = index.search(query_vec, k)

#     # Return top-k matching chunks
#     results = [(texts[i], float(D[0][idx])) for idx, i in enumerate(I[0])]
#     return results

import os
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_query(query):
    return model.encode([query])[0]

def retrieve_top_k(query, k=1, vector_store_dir="Vector_DB"):
    index_path = os.path.join(vector_store_dir, "vector_store.index")
    pkl_path = os.path.join(vector_store_dir, "vector_store_texts.pkl")

    index = faiss.read_index(index_path)

    with open(pkl_path, "rb") as f:
        texts = pickle.load(f)

    query_vec = embed_query(query).astype("float32").reshape(1, -1)
    D, I = index.search(query_vec, k)

    results = [(texts[i], float(D[0][idx])) for idx, i in enumerate(I[0])]
    return results
