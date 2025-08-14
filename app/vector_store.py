import os
import faiss
import pickle
import numpy as np

def save_vector_store(embeddings, texts, output_dir="Vector_DB"):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(output_dir, exist_ok=True)
    # # Save FAISS index
    # faiss.write_index(index, f"{save_path}.index")

    # # Save metadata
    # with open(f"{save_path}_texts.pkl", "wb") as f:
    #     pickle.dump(texts, f)
    faiss.write_index(index, os.path.join(output_dir, "vector_store.index"))
    with open(os.path.join(output_dir, "vector_store_texts.pkl"), "wb") as f:
        pickle.dump(texts, f)

def load_vector_store(output_dir="Vector_DB"):
    index = faiss.read_index(f"{output_dir}/vector_store.index")
    with open(f"{output_dir}/vector_store_texts.pkl", "rb") as f:
        texts = pickle.load(f)
    return index, texts
