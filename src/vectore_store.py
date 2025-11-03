from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from config import OLLAMA_HOST, OLLAMA_EMBED_MODEL, VECTOR_DB_PATH
import os


def create_vector_store(chunks=None, query=None, k=4, persist=True, with_score=False):
  
    # Initialize embeddings
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_HOST
    )
    
      # Create or load vector store
    if chunks:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH 
        )
        print(f"✓ Created vector store with {len(chunks)} chunks")
        if persist:
            print(f"✓ Saved to {VECTOR_DB_PATH}")  # ✅ Fixed f-string
    else:
        if not os.path.exists(VECTOR_DB_PATH):
            print(f"✗ Vector store not found at {VECTOR_DB_PATH}")  # ✅ Fixed f-string
            return None
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embeddings
        )
        print(f"✓ Loaded vector store from {VECTOR_DB_PATH}")  # ✅ Fixed f-string

    # Perform search if a query is given
    if query:
        results = vectorstore.similarity_search(query, k=k)
        print(f"✓ Found {len(results)} chunks")
        return results

    return vectorstore