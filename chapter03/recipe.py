from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

system_prompt = '''\
以下の料理のレシピを考えてください。
出力は、以下のJSON形式で出力してください。
###
{
    "材料": ["材料1", "材料2", "材料3"],
    "手順": ["手順1", "手順2", "手順3"]
}
###

料理名： """
{dish}
"""
'''

def generate_recipe(dish: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"料理名：{dish}"},
        ],
    )
    return response.choices[0].message.content

recipe = generate_recipe("麻婆豆腐")
print(recipe)
