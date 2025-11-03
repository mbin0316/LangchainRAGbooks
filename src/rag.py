from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

from config import *
from vectore_store import create_vector_store


def run_rag(query):
    

    # 1️⃣ Load documents
    loader = DirectoryLoader(DOCS_PATH)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")

    # 2️⃣ Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    # 3️⃣ Build or load vector store using helper function
    print("Creating or loading vector store..")
    vectorstore = create_vector_store(chunks=chunks)
    if not vectorstore:
        raise RuntimeError("Failed to create or load vector store.")

    # 4️ Create retriever from vector store
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 5️ Initialize LLM
    llm = Ollama(base_url=OLLAMA_HOST, model=OLLAMA_MODEL)

    # 6️ Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # 7️ Run user query
    print(f" Query: {query}")
    response = qa_chain.run(query)

    # 8️ Output result
    print("\nResponse:")
    print(response)
    return response

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    run_rag(user_query)

