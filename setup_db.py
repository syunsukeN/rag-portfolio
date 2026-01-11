"""
ChromaDBベクトルデータベースのセットアップスクリプト

このスクリプトは、data/docs/ 内のMarkdownファイルをチャンク分割し、
Gemini Embedding APIでベクトル化してChromaDBに保存します。

v2.0.0の変更点:
- チャンク分割機能の追加（ドキュメント全体 → セクション単位）
- 確認プロンプトの追加（破壊的変更の誤実行防止）
- 進捗表示の改善（どこまで処理したか可視化）
"""

import os
import sys  # 確認プロンプトでsys.exit()を使用
import logging
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# chunker.py から読み込み・分割機能をインポート
from chunker import load_and_chunk_documents

# .envからAPIキーを読み込む
# CHROMA_GOOGLE_GENAI_API_KEY が必要
load_dotenv()

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    メイン処理: ドキュメントをチャンク分割してChromaDBに保存

    処理フロー:
        1. 確認プロンプトで既存データ削除の確認（停止可能性）
        2. ドキュメントをチャンク分割
        3. 既存コレクションを削除
        4. 新規コレクションを作成
        5. チャンクを1件ずつ登録（進捗表示付き）

    期待される結果:
        - 13チャンク（attendance:4, expense:4, it_support:5）
    """
    print("=" * 70)
    print("RAGポートフォリオ - データベースセットアップ v2.0.0")
    print("=" * 70)
    print()

    # ステップ1: 確認プロンプト（停止可能性 - 誤実行防止）
    # データベース再構築は破壊的変更のため、ユーザーに確認を取る
    print("⚠️  データベース構造が変更されます")
    print("   既存のコレクションを削除し、チャンク分割方式で再構築します")
    print("   この操作は元に戻せません")
    print()

    # 環境変数でスキップ可能に（Docker/CI環境での自動実行用）
    # SKIP_CONFIRMATION=true で確認をスキップ
    skip_confirmation = os.getenv("SKIP_CONFIRMATION", "false").lower() == "true"

    if skip_confirmation:
        print("   (SKIP_CONFIRMATION=true のため確認をスキップ)")
    else:
        response = input("   続行しますか? (y/n): ")
        if response.lower() != 'y':
            print("\nキャンセルしました")
            sys.exit(0)  # 正常終了（ユーザーの意図的なキャンセル）

    print()

    # ステップ2: ドキュメントの読み込みとチャンク分割
    print("📂 ドキュメントを読み込み・チャンク分割中...")
    try:
        chunks = load_and_chunk_documents("data/docs")
        print(f"   合計 {len(chunks)} チャンクを作成しました")
    except Exception as e:
        # チャンク生成失敗時は異常終了（停止可能性）
        logger.error(f"チャンク生成失敗: {e}")
        print(f"\n❌ エラー: チャンク生成に失敗しました - {e}")
        sys.exit(1)

    print()

    # ステップ3: ChromaDB の初期化
    print("🗄️  ChromaDBを初期化中...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # Gemini Embedding関数を設定（text-embedding-004を使用）
    # Geminiに送信してベクトル化される（768次元ベクトル）
    gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        model_name="models/text-embedding-004"
    )

    # ステップ4: 既存コレクションの削除（堅牢性 - べき等性を保つ）
    # 既存コレクションがあれば削除（try-except で囲むことで、なければ無視）
    try:
        chroma_client.delete_collection(name="company_docs")
        print("🗑️  既存のコレクションを削除しました")
    except Exception as e:
        # コレクションが存在しない場合のエラーは無視（正常系）
        print("ℹ️  既存のコレクションはありません")
        logger.debug(f"削除時の例外: {e}")  # デバッグログで詳細を記録（改善可能性）

    print()

    # ステップ5: 新規コレクションの作成
    # チャンク分割方式の新しいコレクションを作成
    # embedding_function に Gemini を指定することで、自動的にベクトル化される
    collection = chroma_client.create_collection(
        name="company_docs",
        embedding_function=gemini_ef
    )
    print("📦 新しいコレクションを作成しました")
    print()

    # ステップ6: チャンクの一括登録（改善可能性 - 進捗可視化）
    print("📝 チャンクをベクトル化して保存中...")
    print("   (Gemini Embedding APIでベクトル化するため、少し時間がかかります)")
    print()

    failed_chunks = []  # 失敗したチャンクを記録（改善可能性）

    # 各チャンクを1件ずつ登録（進捗を可視化するため for ループを使用）
    for i, chunk in enumerate(chunks, start=1):
        try:
            # ChromaDBに追加（IDの重複がある場合は上書きされる）
            collection.add(
                ids=[chunk["id"]],
                documents=[chunk["content"]],
                metadatas=[chunk["metadata"]]
            )

            # 進捗ログ（どこまで処理したか分かるよう番号付き）
            # 改善可能性: 後でログから処理状況を確認可能
            print(f"   ✅ [{i:2}/{len(chunks)}] {chunk['metadata']['filename']:20} - {chunk['metadata']['section_title']}")

        except Exception as e:
            # 埋め込み失敗時はログに記録して継続（堅牢性）
            logger.error(f"チャンク登録失敗: {chunk['id']} - {e}")
            failed_chunks.append(chunk['id'])
            print(f"   ❌ [{i:2}/{len(chunks)}] {chunk['id']} - 失敗: {e}")

            # 本番環境では raise して全体を失敗させる選択肢もある（停止可能性）
            # ここでは堅牢性を優先し、一部失敗でも処理を継続
            continue

    print()

    # ステップ7: 完了メッセージ（再現性 - 期待値と比較可能）
    # 最終的なチャンク数を表示し、期待値（13個）と比較できるようにする
    actual_count = collection.count()

    print("=" * 70)
    print("🎉 完了！")
    print("=" * 70)
    print(f"   保存先: ./chroma_db")
    print(f"   チャンク数: {actual_count}")
    # 期待値の説明:
    # - 短いPreambleは最初のH2セクションに結合されるため、独立チャンクにはならない
    # - attendance.md: 3チャンク (勤務時間[+Preamble], 有給休暇, 遅刻・早退)
    # - expense.md: 3チャンク (精算可能な経費[+Preamble], 申請方法, 注意事項)
    # - it_support.md: 4チャンク (パスワード[+Preamble], PC不具合, Wi-Fi, ソフトウェア)
    print(f"   期待値: 10 (attendance:3, expense:3, it_support:4)")

    # 再現性チェック: 期待値と一致するか警告
    if actual_count != 10:
        print(f"\n⚠️  警告: チャンク数が期待値（10）と異なります（実際: {actual_count}）")

    if failed_chunks:
        # 失敗したチャンクがある場合は警告（改善可能性）
        print(f"\n⚠️  警告: {len(failed_chunks)} 個のチャンク登録に失敗しました")
        print(f"   失敗したチャンク: {failed_chunks}")

    print()
    print("データベースの準備ができました✨")
    print("=" * 70)

if __name__ == "__main__":
    main()
