from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
output_parser = StrOutputParser()

cot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザの質問にステップバイステップで回答してください。"),
        ("human", "{question}"),
    ]
)

cot_chain = cot_prompt | model | output_parser

summarize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ステップバイステップで考えた回答から結論だけ抽出してください。"),
        ("human", "{answer}"),
    ]

)

summarize_chain = summarize_prompt | model | output_parser

cot_summarize_chain = cot_chain | summarize_chain
print(cot_summarize_chain.invoke({"question": "10 + 2 * 3 - 5 * 3"}))
