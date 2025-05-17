import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

LANGCHAIN_TRACING="true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
api_key = os.getenv("OPENAI_API_KEY")

# モデルの初期化
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# メッセージの設定
messages = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("こんにちは"),
]

for chunk in model.stream(messages):
    print(chunk.content, end="", flush=True)