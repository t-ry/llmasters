from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[ 
        {"role": "system", "content": "ステップバイステップで考えてください。"},
        {"role": "user", "content": "10 + 2 * 3 - 4 * 2"},
    ],
)

print(response.choices[0].message.content)
