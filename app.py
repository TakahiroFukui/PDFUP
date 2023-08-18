# 必要なモジュールの読み込み
import os
import shutil
from PyPDF2 import PdfReader
import chainlit as cl
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
# 定数設定
OPENAI_API_KEY = "<OpenAI APIキー>"
TEMP_PDF_PATH = "./doc/doc.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 10
DB_PATH = './.chroma'
MODEL_NAME = "gpt-3.5"
# Streamlit Community Cloudの「Secrets」からOpenAI API keyを取得
openai.api_key = st.secrets.OpenAIAPI.openai_api_key
# PDFファイルを開いてテキストを抽出する関数
async def process_pdf(file):
    file = file[0] if isinstance(file, list) else file
    with open(TEMP_PDF_PATH, 'wb') as f:
        f.write(file.content)
    reader = PdfReader(TEMP_PDF_PATH)
    return ''.join(page.extract_text() for page in reader.pages)
# テキストを分割し、埋め込みを作成してデータベースを作成する関数
async def create_db(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = text_splitter.split_text(text)
    metadatas = [{"source": f"{i}-pl"} for i in range(len(docs))]
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_texts(docs, embeddings, metadatas=metadatas)
    return db, docs
# チャットボットの初期化処理
@cl.langchain_factory(use_async=True)
async def init():
    file = None
    while file is None:
        file = await cl.AskFileMessage(content="PDFファイルをアップロードしてください！", accept=["pdf"]).send()
    # データベースの初期化
    shutil.rmtree(DB_PATH, ignore_errors=True)
    # ファイルからテキストを抽出し、データベースを作成
    text = await process_pdf(file)
    db, docs = await create_db(text)
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(model=MODEL_NAME,temperature=0),
        chain_type="stuff",
        retriever=db.as_retriever(),
    )
    # テキストをユーザーセッションに保存
    cl.user_session.set("texts", docs)
    file_name = file[0].name if isinstance(file, list) else file.name
    await cl.Message(content=f"`{file_name}` の準備が完了しました！").send()
    return chain
# 応答を処理する関数
@cl.langchain_postprocess
def process_response(res):
    texts = cl.user_session.get("texts")
    sources = res["sources"].strip().split(',')
    source_elements = [cl.Text(content=texts[int(s[:s.find('-pl')])], name=s) for s in sources if s]
    response = f"{res['answer']} 出典: {res['sources']}"
    cl.Message(content=response, elements=source_elements).send()

if __name__ == '__main__':
    main()
