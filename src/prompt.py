from langchain_core.prompts import ChatPromptTemplate

def chat_prompt():
    prompt = ChatPromptTemplate.from_template(
        """
    You are a helpful AI assistant answering questions about a YouTube video.

    Use ONLY the information provided in the context to answer the user's question.

    If the answer is not available in the context, say:
    "I don't have enough information from the provided video transcript to answer this."

    Do not make up facts or use outside knowledge.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    )

    return prompt