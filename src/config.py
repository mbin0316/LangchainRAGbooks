from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "all-minilm")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vectorstore")

# Get the base directory (project root)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#  docs path
DOCS_PATH = os.path.join(BASE_DIR, "data", "raw")

#full file path
DOC_FILE = os.path.join(DOCS_PATH, "shakespearcompletework.txt")

print("OLLAMA_HOST =", OLLAMA_HOST)
print("TEST_KEY =", OLLAMA_MODEL)
