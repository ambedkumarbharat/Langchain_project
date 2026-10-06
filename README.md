# YouTube RAG Chatbot

A Streamlit app for asking questions about YouTube videos. It retrieves a video's transcript, splits it into text chunks, stores embeddings locally in Chroma, and uses retrieval-augmented generation (RAG) to answer questions from the transcript.

## Features

- Accepts YouTube video URLs.
- Fetches available captions, preferring manually created Hindi or English transcripts when available.
- Splits transcripts into overlapping chunks and creates embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Stores transcript chunks in a persistent local Chroma vector database.
- Uses LangChain's `MultiQueryRetriever` to find relevant transcript context.
- Keeps a chat history while the current Streamlit session is active.

## Requirements

- Python 3.10 or newer
- A valid API key for the OpenAI-compatible chat model endpoint configured in `src/model.py`

The first run may take longer while the Hugging Face embedding model is downloaded.

## Setup

Open a terminal in the project directory and create a virtual environment:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```

On macOS or Linux, use:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root and add your API key:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

Keep the key private. `.env` is excluded from Git; do not commit API keys or other credentials.

## Run the app

From the project root, with the virtual environment active:

```bash
streamlit run app.py
```

In the app, paste a YouTube URL, select **Process Video**, and ask questions in the chat area after processing completes. Videos need to have captions available for transcript retrieval.

There is also a command-line entry point:

```bash
python main.py
```

It prompts for a YouTube URL and a question.

## Project structure

```text
.
├── app.py                  # Streamlit user interface
├── main.py                 # Command-line example
├── requirements.txt        # Python dependencies
├── src/
│   ├── embedding.py        # Hugging Face embedding model
│   ├── model.py            # OpenAI-compatible chat model
│   ├── prompt.py           # Context-grounded answer prompt
│   ├── retriever.py        # Multi-query retrieval
│   ├── splitter.py         # Transcript chunking
│   ├── transcript_loader.py # YouTube transcript retrieval
│   ├── url_to_id.py        # YouTube URL and video ID parsing
│   └── vectorstore.py      # Chroma storage and chunk indexing
└── data/
    └── chroma_db/          # Local persistent vector database
```

## Data and answers

The Chroma database is stored locally under `data/chroma_db/` and is ignored by Git. The app answers from retrieved transcript context and is instructed to say when that context does not contain enough information. Questions and retrieved context are sent to the configured chat model provider to generate answers.
