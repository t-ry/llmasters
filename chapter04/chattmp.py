import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

LANGCHAIN_TRACING="true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("chat_history", optional=True),
    ("user", "{input}"),
    ]
)

prompt_value = prompt.invoke(
    {
        "chat_history": [
            HumanMessage(content="こんにちは！私は山浦翼と言います！"),
            AIMessage(content="どのようにお手伝いできますか？"),
        ],
        "input": "私のあだ名を考えてください！",
    }
)

# モデルを呼び出して応答を取得
ai_message = model.invoke(prompt_value)

# 応答を表示
print("AIの応答:", ai_message.content)
