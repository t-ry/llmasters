from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "質問に100文字程度で答えてください"},
        {"role": "user", "content": "プロンプトエンジニアリングって何？"},
    ],
)


print(response.choices[0].message.content)