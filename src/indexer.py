import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

import config


def build_index() -> VectorStoreIndex:
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = client.get_or_create_collection(config.CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    if collection.count() > 0:
        print(f"Chunks indexed: {collection.count()}")
        return VectorStoreIndex.from_vector_store(vector_store)

    documents = SimpleDirectoryReader(str(config.DATA_DIR)).load_data()
    print(f"Loaded {len(documents)} documents")
    return VectorStoreIndex.from_documents(documents, storage_context=storage_ctx)