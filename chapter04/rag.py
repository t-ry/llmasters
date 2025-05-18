from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


# 環境変数の読み込み（.envファイルからAPIキーなどを読み込む）
load_dotenv()

# GPT-4モデルの初期化（temperature=0で一貫性のある出力を確保）
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# Gitリポジトリから読み込むファイルのフィルタリング関数
# .mdxファイルのみを対象とする
def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


# LangChainのGitHubリポジトリからドキュメントを読み込む
loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)

# ドキュメントの読み込み
raw_docs = loader.load()
#print(len(raw_docs))

# テキストを適切なサイズに分割する設定
# chunk_size: 各チャンクの最大文字数
# chunk_overlap: チャンク間の重複文字数
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# ドキュメントの分割
docs = text_splitter.split_documents(raw_docs)
#print(len(docs))

# テキストの埋め込みベクトルを生成するためのモデルを初期化
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 検索クエリの設定
query = "AWSのS3からデータを読み込むためのDocument loaderはありますか？"

# Chromaベクトルデータベースの作成とドキュメントをベクトル化し、保存
db = Chroma.from_documents(docs, embeddings)

# 検索機能の初期化
retriever = db.as_retriever()

# クエリに基づいて関連ドキュメントを検索
context_docs = retriever.invoke(query)
print(f"len = {len(context_docs)}")

# 最初の検索結果の詳細を表示
first_doc = context_docs[0]
print(f"metadata = {first_doc.metadata}")
print(first_doc.page_content)


# クエリの埋め込みベクトルを生成する例（コメントアウト）
vector = embeddings.embed_query(query)
print(len(vector))
print(vector)


