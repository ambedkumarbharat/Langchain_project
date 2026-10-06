# YouTube RAG

A Retrieval-Augmented Generation (RAG) system for querying YouTube video transcripts using LangChain, ChromaDB, and Streamlit.

## Project Structure

```text
youtube-rag/
│
├── app.py                         # Streamlit UI
├── requirements.txt
├── .env
├── .gitignore
├── README.md
│
├── data/
│   ├── transcripts/               # YouTube transcripts
│   └── chroma_db/                 # Local vector database
│
├── src/
│   ├── __init__.py
│   │
│   ├── youtube/
│   │   ├── __init__.py
│   │   └── loader.py              # YouTube URL → transcript
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── splitter.py            # Transcript → chunks
│   │   └── embedder.py            # Chunks → embeddings
│   │
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── chroma.py              # Chroma DB
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py           # Similarity search
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── prompt.py              # RAG prompt
│   │   └── chain.py               # Retriever + LLM
│   │
│   └── utils/
│       ├── __init__.py
│       └── youtube.py              # URL/video ID utilities
│
└── tests/
    ├── test_youtube.py
    ├── test_splitter.py
    └── test_retrieval.py
```

## Setup & Installation

1. Activate the virtual environment:
   - **Windows**: `.\venv\Scripts\Activate.ps1` (or `.\venv\Scripts\activate.bat`)
   - **Linux/macOS**: `source venv/bin/activate`

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   - Update `.env` with your API keys.

4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
