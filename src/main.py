#from src.rag import run_rag
from config import *
from rag import *

def load_documents():
    with open(DOC_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    print("Preview:\n", content[:300])
    return content

if __name__ == "__main__":
    load_documents()
    query = "What is the main summary  of book you read?"
    run_rag(query)
    
   
