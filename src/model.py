from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    model = ChatOpenAI(
        base_url="https://api.xkiro.com/v1",
        model="qwen/qwen3.8-omni-flash:free"
    )

    return model

