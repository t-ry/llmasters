# 必要なライブラリのインポート
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import GitLoader


# プロンプトテンプレートの定義
# 文脈と質問を受け取り、文脈に基づいて回答を生成する形式
prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて、質問に回答してください。

文脈："""
{context}
"""
                                          
質問：{question}
''')


# Gitリポジトリから読み込むファイルのフィルタリング関数
# .mdxファイルのみを対象とする
def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")

# 環境変数の読み込み（.envファイルからAPIキーなどを読み込む）
load_dotenv()

# GPT-4モデルの初期化（temperature=0で一貫性のある出力を確保）
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# LangChainのGitHubリポジトリからドキュメントを読み込む
loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)

# ドキュメントの読み込み
raw_docs = loader.load()

# テキストを適切なサイズに分割する設定
# chunk_size: 各チャンクの最大文字数
# chunk_overlap: チャンク間の重複文字数
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# ドキュメントの分割
docs = text_splitter.split_documents(raw_docs)

# テキストの埋め込みベクトルを生成するためのモデルを初期化
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Chromaベクトルデータベースの作成とドキュメントをベクトル化し、保存
db = Chroma.from_documents(docs, embeddings)

# 検索機能の初期化
retriever = db.as_retriever()

# LCEL（LangChain Expression Language）を使用したチェーンの構築
# 1. コンテキストの取得（retriever）と質問の受け渡し（RunnablePassthrough）
# 2. プロンプトテンプレートの適用
# 3. モデルによる回答生成
# 4. 文字列として出力をパース
chain = (
    {"context":retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# 検索クエリの設定
query = "AWSのS3からデータを読み込むためのDocument loaderはありますか？"

# チェーンの実行と結果の出力
output = chain.invoke(query)
print(output)

