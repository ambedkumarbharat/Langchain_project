"""Transcript text splitter module."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_transcript(
    transcript: dict,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
):
    """Convert transcript data into chunks."""

    document = Document(
        page_content=transcript["page_content"],
        metadata={
            "video_id": transcript["video_id"],
            "language": transcript["language"],
            "is_generated": transcript["is_generated"],
        },
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    return splitter.split_documents([document])