from src.transcript_loader import get_transcript
from src.splitter import split_transcript
from src.model import get_llm
from src.prompt import chat_prompt
from src.retriever import get_multi_query_retriever
from src.vectorstore import get_vector, add_chunks

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
)


parser = StrOutputParser()

youtube_url = input("Paste your URL: ")
query = input("Ask question around your video: ")


# -----------------------------
# Vector Store
# -----------------------------

vectorstore = get_vector()


# -----------------------------
# Ingestion Runnables
# -----------------------------

transcript = RunnableLambda(get_transcript)

splitter = RunnableLambda(split_transcript)

add_to_vectorstore = RunnableLambda(
    lambda chunks: add_chunks(
        vectorstore,
        chunks,
        chunks[0].metadata["video_id"],
    )
)


# -----------------------------
# YouTube → Vector Store
# -----------------------------

script_chain = (transcript | splitter | add_to_vectorstore)

script_chain.invoke(youtube_url)


# -----------------------------
# Retrieval
# -----------------------------

retriever = get_multi_query_retriever(vectorstore)


parallel_chain = RunnableParallel(
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
)


# -----------------------------
# RAG Chain
# -----------------------------

prompt = chat_prompt()
llm = get_llm()

chain = ( parallel_chain | prompt | llm | parser)


# -----------------------------
# Final Answer
# -----------------------------

answer = chain.invoke(query)

print("\nAnswer:\n")
print(answer)