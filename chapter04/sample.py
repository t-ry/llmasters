import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()

LANGCHAIN_TRACING="true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
api_key = os.getenv("OPENAI_API_KEY")


model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage("こんにちは！私は山浦翼と言います！"),
    AIMessage(content="こんにちは、山浦翼さん！どのようにお手伝いできますか？"),
    HumanMessage(content="私の名前がわかりますか？"),
]

ai_message = model.invoke(messages)
print(ai_message.content)









