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
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage("こんにちは！私は山浦翼と言います！"),
    AIMessage(content="こんにちは、山浦ツバサさん！どのようにお手伝いできますか？"),
    HumanMessage(content="私の名前がわかりますか？"),
]

# モデルの実行と結果の表示
ai_message = model.invoke(messages)
print(ai_message.content)