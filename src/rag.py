from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import *
from langchain_community.document_loaders import DirectoryLoader

def run_rag(query):
        loader=DirectoryLoader(DOCS_PATH)
        docs= loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        text=splitter.split_documents(docs)

        print(f"Loaded {len(docs)} documents")
        print(f"Split into {len(text)} chunks")
    
        return text