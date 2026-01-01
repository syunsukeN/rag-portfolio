import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# .envからAPIキーを読み込む
load_dotenv()

# ドキュメントを読み込む関数
def load_documents(docs_path):
    documents = []
    for filename in os.listdir(docs_path):
        if filename.endswith(".md"):
            filepath = os.path.join(docs_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append({
                    "id": filename,
                    "content": content,
                    "filename": filename
                })
    return documents

# メイン処理
def main():
    print("📂 ドキュメントを読み込み中...")
    docs = load_documents("data/docs")
    print(f"   {len(docs)}件のドキュメントを読み込みました")

    # ChromaDBの設定
    print("🗄️  ChromaDBを初期化中...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # Gemini Embedding関数を設定（text-embedding-004を使用）
    gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        model_name="models/text-embedding-004"
    )

    # コレクション（テーブルのようなもの）を作成
    collection = chroma_client.get_or_create_collection(
        name="company_docs",
        embedding_function=gemini_ef
    )

    # ドキュメントを追加
    print("📝 ドキュメントをベクトル化して保存中...")
    for doc in docs:
        collection.upsert(
            ids=[doc["id"]],
            documents=[doc["content"]],
            metadatas=[{"filename": doc["filename"]}]
        )
        print(f"   ✅ {doc['filename']} を保存しました")

    print("🎉 完了！データベースの準備ができました")
    print(f"   保存先: ./chroma_db")
    print(f"   ドキュメント数: {collection.count()}")

if __name__ == "__main__":
    main()
