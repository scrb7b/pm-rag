from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / ".chroma"

CHROMA_COLLECTION = "pm-rag"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"
LLM_TIMEOUT = 120
LLM_CONTEXT_WINDOW = 4096
TOP_K = 3
