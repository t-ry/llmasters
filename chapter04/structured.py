from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# 環境変数の読み込み
load_dotenv()

# モデルの設定
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# クラスの定義
class Recipe(BaseModel):
    ingredients: list[str] = Field(description="ingredients of the dish")
    steps: list[str] = Field(description="steps to make the dish")
    


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザーが入力した料理のレシピを考えてください。"),
        ("human", "{dish}"),
    ]
)

chain = prompt | model.with_structured_output(Recipe)

recipe = chain.invoke({"dish": "チーズケーキ"}) 
print(type(recipe))
print(recipe)


