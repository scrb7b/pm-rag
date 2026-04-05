import sys
from pathlib import Path

import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = Path(__file__).parent / ".chroma"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"


def build_index() -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection("pm-rag")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    if collection.count() > 0:
        print(f"Using cached index ({collection.count()} chunks).")
        return VectorStoreIndex.from_vector_store(vector_store)

    print("Building index from documents...")
    documents = SimpleDirectoryReader(str(DATA_DIR)).load_data()
    print(f"Loaded {len(documents)} documents.")

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_ctx)
    return index


def main():
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=120, context_window=4096)

    index = build_index()
    engine = index.as_query_engine(similarity_top_k=3)

    print(f"\nReady. LLM: {LLM_MODEL}, embed: {EMBED_MODEL}.\n")

    while True:
        try:
            question = input("Q: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not question:
            continue

        response = engine.query(question)

        print(f"\nA: {response}\n")
        print("-" * 60)

if __name__ == "__main__":
    main()
