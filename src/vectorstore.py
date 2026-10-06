import hashlib

from langchain_chroma import Chroma
from .embedding import get_embeddings


def generate_chunk_id(video_id: str, chunk_number: int, content: str) -> str:
    """Generate a stable unique ID for each chunk."""

    text = f"{video_id}_{chunk_number}_{content}"

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_vector():
    """Return Chroma vector store."""

    vectorstore = Chroma(
        persist_directory="data/chroma_db",
        embedding_function=get_embeddings(),
        collection_name="youtube_transcripts",
    )

    return vectorstore


def add_chunks(vectorstore, chunks, video_id):
    """Add only new chunks to Chroma."""

    existing_ids = set(vectorstore.get()["ids"])

    new_chunks = []
    new_ids = []

    for chunk_number, chunk in enumerate(chunks):

        chunk_id = generate_chunk_id(
            video_id=video_id,
            chunk_number=chunk_number,
            content=chunk.page_content,
        )

        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(chunk_id)

    if new_chunks:
        vectorstore.add_documents(
            documents=new_chunks,
            ids=new_ids,
        )

        print(f"Added {len(new_chunks)} new chunks.")

    else:
        print("No new chunks. Already exists.")
