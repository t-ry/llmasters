from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)



prompt = ChatPromptTemplate.from_messages   (
    [
        ("system", "ユーザーが入力したスポーツのルールを簡素に教えてください。"),
        ("human", "{input}"),
    ]
)

output_parser = StrOutputParser()

# チェーンの構築
chain = prompt | model | output_parser

# ストリーミング出力
#for chunk in chain.stream({"input": "サッカー"}) :
#    print(chunk, end="", flush=True) 

# バッチ処理の並列実行
outputs = chain.batch([{"input": "サッカー"}, {"input": "バスケットボール"}])
print(outputs)