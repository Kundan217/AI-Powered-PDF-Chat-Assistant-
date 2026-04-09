import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
TOP_K_RESULTS = 4

# Models
LLM_MODEL = "models/gemini-2.5-flash"        # valid Gemini chat model for current API version
EMBEDDING_MODEL = "models/gemini-embedding-001"  # free Gemini embeddings