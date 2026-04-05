# pm-rag

Minimal local RAG: ask questions, get answers from your docs.

**Stack:** Ollama (`llama3.2` + `nomic-embed-text`), ChromaDB, LlamaIndex

## Requirements

- [Ollama](https://ollama.com) running locally
- Python 3.12+

Pull the models before first run:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Install & run

**With uv (recommended):**

```bash
uv sync
uv run python main.py
```

**With pip:**

```bash
pip install -r requirements.txt
python main.py
```

## Project structure

```
pm-rag/
├── src/
│   ├── config.py    # paths, model names
│   ├── indexer.py   # index build 
│   └── main.py      
├── data/            # put your source documents here
└── .chroma/         # vector store
```

Put your documents in `data/`. Index is built on first run and cached in `.chroma/` — subsequent runs start instantly.

## Example

```
Q: How does the product work?
A: It collects them using sensors and computer vision, washes them internally,
   and then returns them clean.

Q: Can I disable jokes?
A: Yes, you can disable jokes in your RoboDish by using entertainment mode in settings.

Q: RoboDish 3000™
A: RoboDish 3000 is a mobile robotic dishwasher designed for home and office use.
   It autonomously moves around, collects dirty dishes, cleans them, and returns
   them to designated areas, while also providing entertainment through jokes and
   songs after completing its cleaning cycle.
```
