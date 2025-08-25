import numpy as np
from app.chunker import chunk_text
from app.embedder import embed_texts
from app.vector_store import save_vector_store
from app.retriever import retrieve_top_k
from app.llm import generate_answer



def load_raw_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    # Step 1: Load your data
    text = load_raw_text("data/sample.txt")  # Use a PDF/text loader if needed //
    print("📄 Data loaded successfully.")
    print(text)

    # Step 2: Chunk it
    chunks = chunk_text(text) 
    print(f"📦 Data chunked into {len(chunks)} pieces.")
    print(chunks)

    # Step 3: Embed it
    embeddings = embed_texts(chunks) 
    print("🔍 Texts embedded successfully.")
    print(f"Embeddings shape: {embeddings.shape}")

    # Step 4: Save it in FAISS
    save_vector_store(np.array(embeddings), chunks) 

    print("✅ Vector store created and saved.")

    while True:
        query = input("\n🔎 Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        results = retrieve_top_k(query)
        context = "\n".join([text for text, _ in results])

        print("\n🤖 Generating answer...\n")
        context='temp'
        response = generate_answer(context, query)
        print(response)
