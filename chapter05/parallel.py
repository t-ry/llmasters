from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv
import pprint

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
output_parser = StrOutputParser()

optimistic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは、楽観主義者です。ユーザーからの話題に対して楽観的な意見をください。"),
        ("human", "{topic}"),
    ]
)


optimistic_chain = optimistic_prompt | model | output_parser

pessimistic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは、悲観主義者です。ユーザーからの話題に対して、悲観的な意見をください。"),
        ("human", "{topic}"),
    ]
)

pessimistic_chain = pessimistic_prompt | model | output_parser

parallel_chain = RunnableParallel(
    {
        "optimistic_opinion": optimistic_chain,
        "pessimistic_opinion": pessimistic_chain,
    }
)

output = parallel_chain.invoke({"topic": "生成AIの進化について"})
pprint.pprint(output)

synthesize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは客観的AIです。2つの意見をまとめてください。"),
        ("human", "楽観的意見:{optimistic_opinion}\n悲観的意見:{pessimistic_opinion}"),
    ]
)

synthesize_chain = parallel_chain | synthesize_prompt | model | output_parser

output = synthesize_chain.invoke({"topic": "生成AIの進化について"})
print(output)




