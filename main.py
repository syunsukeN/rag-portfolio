import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# .envから環境変数を読み込む
load_dotenv()

app = FastAPI()

# CORS設定を追加（フロントエンドからのアクセスを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切に制限する
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Gemini APIの設定
genai.configure(api_key=os.getenv("CHROMA_GOOGLE_GENAI_API_KEY"))

# ChromaDBの設定
chroma_client = chromadb.PersistentClient(path="./chroma_db")
gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    model_name="models/text-embedding-004"
)
collection = chroma_client.get_or_create_collection(
    name="company_docs",
    embedding_function=gemini_ef
)

# リクエストの型を定義（Flutterのクラスに似ている）
class Question(BaseModel):
    text: str

# ルートエンドポイント
@app.get("/")
def read_root():
    return {"message": "RAGポートフォリオへようこそ！"}

# ヘルスチェック
@app.get("/health")
def health_check():
    return {"status": "ok"}

# 検索エンドポイント
@app.post("/search")
def search_documents(question: Question):
    # 質問に関連するドキュメントを検索
    results = collection.query(
        query_texts=[question.text],
        n_results=2  # 上位2件を取得
    )

    return {
        "question": question.text,
        "results": [
            {
                "filename": meta["filename"],
                "content": doc,
                "distance": dist
            }
            for meta, doc, dist in zip(
                results["metadatas"][0],
                results["documents"][0],
                results["distances"][0]
            )
        ]
    }

# 質問応答エンドポイント（RAG）
@app.post("/ask")
def ask_question(question: Question):
    # 1. 関連ドキュメントを検索
    results = collection.query(
        query_texts=[question.text],
        n_results=2
    )

    # 2. 検索結果をコンテキストとして整形
    context = "\n\n---\n\n".join(results["documents"][0])

    # 3. プロンプトを作成
    system_prompt = """あなたは社内アシスタントです。
以下のルールを必ず守ってください：

1. 提供された社内資料のみを参照して回答してください
2. 資料に書かれていない情報は「その情報は社内資料に記載がありません」と回答してください
3. 回答の根拠となる資料名を必ず示してください
4. 簡潔に、箇条書きで回答してください"""

    user_prompt = f"""## 社内資料
{context}

## 質問
{question.text}

上記の社内資料を参照して回答してください。"""

    # 4. Gemini APIで回答生成
    model = genai.GenerativeModel('models/gemini-2.5-flash-preview-09-2025')

    # システムプロンプトとユーザープロンプトを結合
    full_prompt = f"""{system_prompt}

{user_prompt}"""

    response = model.generate_content(
        full_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,  # 低めに設定して正確性を重視
        )
    )

    answer = response.text

    # 5. 結果を返す
    return {
        "question": question.text,
        "answer": answer,
        "sources": [meta["filename"] for meta in results["metadatas"][0]]
    }