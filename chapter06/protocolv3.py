from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
#from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings

# 環境変数の読み込み
load_dotenv()

# モデルの指定
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# PDFファイルのURLを指定
pdf_url = "https://www.fdma.go.jp/mission/enrichment/appropriate/items/119_tuhou.pdf"

# PDFファイルの読み込み
loader = PyPDFLoader(pdf_url)
documents = loader.load()
print(len(documents))

# テキストの埋め込みベクトルを生成するためのモデルを初期化
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# Chromaデータベースにドキュメントと埋め込みベクトルを保存
db = Chroma.from_documents(documents, embeddings)

# 質問応答のためのプロンプトテンプレートを定義
prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問:{question}
''')

# データベースから関連文脈を検索するためのリトリーバーを作成
retriever = db.as_retriever()

# 処理チェーンの構築
# 1. 質問を受け取り
# 2. 関連文脈を検索
# 3. プロンプトテンプレートに適用
# 4. GPTモデルで回答生成
# 5. 文字列として出力
chain = {
    "question": RunnablePassthrough(),
    "context": retriever,
} | prompt | model | StrOutputParser()

# チェーンを実行して質問に回答
print(chain.invoke("プロトコルバージョン2との違いは何？"))