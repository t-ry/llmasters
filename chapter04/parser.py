from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 環境変数の読み込み
load_dotenv()

# モデルの設定
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# モデルの出力形式を指定
class Recipe(BaseModel):
    ingredients: list[str] = Field(description="ingredients of the dish")
    steps: list[str] = Field(description="steps to make the dish")

# モデルの出力形式を指定
output_parser = PydanticOutputParser(pydantic_object=Recipe)

# モデルの出力形式を指定
format_instructions = output_parser.get_format_instructions()
print(format_instructions)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "ユーザーが入力した料理のレシピを教えてください。\n\n"
            "{format_instructions}",
        ),
        ("human", "{dish}"),
    ]
)

prompt_with_format_instructions = prompt.partial(
    format_instructions=format_instructions
)

prompt_value = prompt_with_format_instructions.invoke({"dish": "チーズケーキ"})
print("====role: system=====")
print(prompt_value.messages[0].content)
print("====role: user=====")
print(prompt_value.messages[1].content)

ai_message = model.invoke(prompt_value)
print("====role: ai=====")
print(ai_message.content)


recipe = output_parser.parse(ai_message.content)
print(type(recipe))
print(recipe)   










