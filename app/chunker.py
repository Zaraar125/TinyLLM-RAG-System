from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text, chunk_size=50, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)
