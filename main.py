from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

import config
from indexer import build_index


def main():
    Settings.embed_model = OllamaEmbedding(model_name=config.EMBED_MODEL)
    Settings.llm = Ollama(
        model=config.LLM_MODEL,
        request_timeout=config.LLM_TIMEOUT,
        context_window=config.LLM_CONTEXT_WINDOW,
    )

    engine = build_index().as_query_engine(similarity_top_k=config.TOP_K)
    print(f"LLM: {config.LLM_MODEL}, embedder: {config.EMBED_MODEL}")

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
