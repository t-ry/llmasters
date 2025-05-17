from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 環境変数の読み込み
load_dotenv()


# クラスの定義
class Recipe(BaseModel):
    ingredients: list[str] = Field(description="ingredients of the dish")
    steps: list[str] = Field(description="steps to make the dish")

# 出力形式の指定
output_parser = PydanticOutputParser(pydantic_object=Recipe)

# プロンプトの作成
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザが入力した料理のレシピを考えてください。\n\n{format_instructions}"),
        ("human", "{dish}"),
    ]
)

prompt_with_format_instructions = prompt.partial(
    format_instructions = output_parser.get_format_instructions()
)

# モデルの設定&JSON modeの指定
model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind(
    response_format={"type": "json_object"}
)

chain = prompt_with_format_instructions | model | output_parser

recipe = chain.invoke({"dish": "チーズケーキ"})
print(type(recipe))
print(recipe)


