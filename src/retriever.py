from langchain_classic.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from .model import get_llm


def get_multi_query_retriever(vectorstore):

    llm = get_llm()

    base_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )