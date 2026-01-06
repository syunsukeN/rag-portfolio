"""
RAGベースの社内資料検索AIのバックエンドAPI

v2.1.0の主な変更点:
- クエリ拡張機能の追加（LLMによる検索キーワード拡張）
- レスポンスに expanded_query, added_keywords を追加
- 環境変数 QUERY_EXPANSION_ENABLED で機能の有効/無効を制御可能

v2.0.0の主な変更点:
- チャンク分割対応（ドキュメント全体 → セクション単位の検索）
- 信頼度スコア（confidence）の追加
- セクション情報の詳細化
- 検索件数の最適化（n_results: 2 → 3）

技術スタック:
- FastAPI: RESTful API フレームワーク
- ChromaDB: ベクトルデータベース（類似度検索）
- Gemini API: 埋め込み生成とLLM応答生成
- QueryExpander: クエリ拡張モジュール（v2.1.0で追加）
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# クエリ拡張モジュールをインポート（v2.1.0で追加）
from query_expander import QueryExpander

# .envから環境変数を読み込む
# 必要な環境変数: CHROMA_GOOGLE_GENAI_API_KEY
load_dotenv()

# v2.1.0: クエリ拡張機能を追加
# 変更点: レスポンスに expanded_query, added_keywords フィールドを追加
app = FastAPI(
    title="社内資料検索AI",
    description="RAGベースの質問応答システム - クエリ拡張対応",
    version="2.1.0"
)

# CORS設定を追加（フロントエンドからのアクセスを許可）
# 本番環境では allow_origins を適切に制限してください
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 全オリジンを許可（開発用）
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Gemini APIの設定
# 埋め込み（Embedding）とLLM応答の両方で使用
genai.configure(api_key=os.getenv("CHROMA_GOOGLE_GENAI_API_KEY"))

# ChromaDBの設定
# PersistentClient: ディスクに永続化（./chroma_db/）
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Gemini Embedding関数を設定
# text-embedding-004: 768次元ベクトルを生成
gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    model_name="models/text-embedding-004"
)

# コレクションを取得（v2.0.0ではチャンク分割されたデータ）
collection = chroma_client.get_or_create_collection(
    name="company_docs",
    embedding_function=gemini_ef
)

# クエリ拡張機能の初期化（v2.1.0で追加）
# 環境変数 QUERY_EXPANSION_ENABLED=false で無効化可能
# デフォルトは有効（true）
query_expander = QueryExpander(
    enabled=os.getenv("QUERY_EXPANSION_ENABLED", "true").lower() == "true"
)


# リクエストの型を定義（Pydantic BaseModel）
class Question(BaseModel):
    """
    質問リクエストの型定義

    Attributes:
        text (str): ユーザーの質問文
    """
    text: str


def calculate_confidence(results) -> str:
    """
    類似度スコアに基づいて信頼度レベルを計算

    Args:
        results: ChromaDB の query() の戻り値
            - documents: 取得したドキュメント
            - distances: 各ドキュメントとクエリの距離

    Returns:
        str: 信頼度レベル
            - "high": 0.8以上（ほぼ完全一致）
            - "medium": 0.6以上（関連性あり）
            - "low": 0.6未満（関連性が薄い）
            - "none": 検索結果なし

    説明:
        ChromaDBは距離（distance）を返す（小さいほど類似）
        類似度 = 1.0 - 距離 で変換（大きいほど類似）

        閾値の根拠（CLAUDE.mdで合意）:
        - high >= 0.8: ユーザーが求める情報とほぼ一致
        - medium >= 0.6: 関連性はあるが、完全一致ではない
        - low < 0.6: 関連性が薄い、または情報不足

        実務での使い方:
        - Phase 2で評価用質問セットを用意し、Recall@k で閾値をチューニング
        - 低信頼度クエリをログに記録し、ドキュメント改善に活用
    """
    # 検索結果が空の場合
    if not results["documents"] or not results["documents"][0]:
        return "none"

    # ChromaDB の距離を類似度に変換
    # 距離: 0に近い = 類似、1に近い = 非類似（コサイン距離）
    top_distance = results["distances"][0][0]
    top_similarity = 1.0 - top_distance  # 距離の逆数で類似度を計算

    # 閾値で信頼度レベルを判定（説明可能性 - なぜこの閾値か）
    if top_similarity >= 0.8:
        return "high"  # 高信頼: ほぼ完全一致
    elif top_similarity >= 0.6:
        return "medium"  # 中信頼: 関連性あり
    else:
        return "low"  # 低信頼: 関連性が薄い


# ルートエンドポイント
@app.get("/")
def read_root():
    """
    ルートエンドポイント - APIの基本情報を返す

    Returns:
        dict: メッセージとバージョン情報

    使用例:
        curl http://localhost:8000/
    """
    return {
        "message": "RAGポートフォリオ v2.1.0 - クエリ拡張対応",
        "version": "2.1.0",
        "endpoints": {
            "/health": "ヘルスチェック",
            "/search": "ベクトル検索のみ（LLMなし）",
            "/ask": "質問応答（RAG）"
        }
    }


# ヘルスチェックエンドポイント
@app.get("/health")
def health_check():
    """
    ヘルスチェック - システムの稼働状況を確認

    Returns:
        dict: ステータスとチャンク数、バージョン、クエリ拡張の状態

    使用例:
        curl http://localhost:8000/health

    期待される出力:
        {"status": "ok", "total_chunks": 10, "query_expansion_enabled": true, "version": "2.1.0"}
    """
    return {
        "status": "ok",
        "total_chunks": collection.count(),
        "query_expansion_enabled": query_expander.enabled,  # v2.1.0で追加
        "version": "2.1.0"
    }


# 検索エンドポイント（LLMなし）
@app.post("/search")
def search_documents(question: Question):
    """
    ベクトル検索のみ実行（LLM生成なし）

    Args:
        question (Question): 検索クエリ

    Returns:
        dict: 検索結果（上位3件）
            - question: 検索クエリ
            - expanded_query: 拡張後のクエリ（v2.1.0で追加）
            - expansion_applied: 拡張が適用されたか（v2.1.0で追加）
            - results: 検索結果リスト
                - filename: ファイル名
                - section: セクションタイトル
                - content: ドキュメント内容
                - similarity_score: 類似度スコア（0-1、高いほど類似）

    使用例:
        curl -X POST http://localhost:8000/search \
          -H "Content-Type: application/json" \
          -d '{"text": "有給休暇について"}'

    v2.1.0の変更点:
        - クエリ拡張機能の追加（LLMでドメイン固有キーワードを追加）
        - expanded_query, expansion_applied フィールドの追加

    v2.0.0の変更点:
        - n_results: 2 → 3（チャンク分割により件数増加）
        - section フィールドの追加（セクションタイトル）
        - similarity_score の追加（1 - distance）
    """
    # ステップ1: クエリ拡張を適用（v2.1.0で追加）
    # LLMでドメイン固有のキーワードを追加し、検索精度を向上
    expansion_result = query_expander.expand(question.text)

    # ステップ2: 拡張後のクエリでベクトル検索
    # チャンク分割により各セクションが小さくなったため、検索件数を3に設定
    results = collection.query(
        query_texts=[expansion_result.expanded_query],  # 拡張後のクエリを使用
        n_results=3
    )

    # レスポンスの構築（クエリ拡張情報を含む）
    return {
        "question": question.text,
        "expanded_query": expansion_result.expanded_query,  # v2.1.0で追加
        "expansion_applied": expansion_result.expansion_applied,  # v2.1.0で追加
        "results": [
            {
                "filename": meta["filename"],
                "section": meta["section_title"],
                "content": doc,
                "similarity_score": round(1 - dist, 3)
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
    """
    質問応答エンドポイント - RAG（Retrieval-Augmented Generation）

    処理フロー:
        1. クエリ拡張（LLMでドメイン固有キーワードを追加）← v2.1.0で追加
        2. ベクトル検索で関連チャンクを取得（上位3件）
        3. 信頼度レベルを計算（high/medium/low）
        4. 検索結果をコンテキストとして整形（セクション情報付き）
        5. LLMプロンプトを構築
        6. Gemini APIで回答を生成
        7. 回答とメタデータを返却

    Args:
        question (Question): ユーザーの質問

    Returns:
        dict: 質問応答の結果
            - question: 質問文
            - expanded_query: 拡張後のクエリ（v2.1.0で追加）
            - added_keywords: 追加されたキーワード（v2.1.0で追加）
            - answer: LLMが生成した回答
            - sources: 参照したファイル名リスト（後方互換性）
            - sections: 詳細なセクション情報（v2.0.0で追加）
            - confidence: 信頼度レベル（v2.0.0で追加）
            - version: APIバージョン

    使用例:
        curl -X POST http://localhost:8000/ask \
          -H "Content-Type: application/json" \
          -d '{"text": "有給休暇は何日ですか？"}'

    期待される出力:
        {
          "question": "有給休暇は何日ですか？",
          "expanded_query": "有給休暇は何日ですか？ 有給 勤怠 休み",
          "added_keywords": ["有給", "勤怠", "休み"],
          "answer": "...",
          "sources": ["attendance.md"],
          "sections": [
            {"filename": "attendance.md", "section_title": "有給休暇", "similarity_score": 0.85}
          ],
          "confidence": "high",
          "version": "2.1.0"
        }

    v2.1.0の変更点:
        - クエリ拡張機能の追加（LLMでドメイン固有キーワードを追加）
        - expanded_query, added_keywords フィールドの追加

    v2.0.0の変更点:
        - n_results: 2 → 3（検索精度向上）
        - セクション情報をコンテキストに含める
        - プロンプト強化（セクション名の引用を指示）
        - 信頼度レベルの計算
        - レスポンスに sections, confidence, version を追加
    """
    # ステップ0: クエリ拡張を適用（v2.1.0で追加）
    # LLMでドメイン固有のキーワードを追加し、検索精度を向上
    # 例: "有給の申請方法" → "有給の申請方法 有給休暇 勤怠 休み"
    expansion_result = query_expander.expand(question.text)

    # ステップ1: 拡張後のクエリで関連チャンクを検索
    # チャンク分割により各セクションが小さくなったため、検索件数を3に設定
    results = collection.query(
        query_texts=[expansion_result.expanded_query],  # 拡張後のクエリを使用
        n_results=3
    )

    # ステップ2: 信頼度レベルを計算
    confidence = calculate_confidence(results)

    # ステップ3: 検索結果をコンテキストとして整形（セクション情報を含む）
    # 各チャンクに「ファイル名 - セクション名」を明示
    context_parts = []
    for meta, doc in zip(results["metadatas"][0], results["documents"][0]):
        context_parts.append(
            f"【{meta['filename']} - {meta['section_title']}】\n{doc}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # ステップ4: プロンプトを構築
    # v2.0.0の改善: セクション情報の引用を明示的に指示
    system_prompt = """あなたは社内アシスタントです。
以下のルールを必ず守ってください:

1. 提供された社内資料の特定セクションのみを参照して回答してください
2. 資料に書かれていない情報は「その情報は社内資料に記載がありません」と回答してください
3. 回答の根拠となる資料名とセクション名を必ず示してください（例: attendance.md - 有給休暇）
4. 簡潔に、箇条書きで回答してください
5. 複数のセクションから情報を統合する場合は、それぞれの出典を明記してください"""

    user_prompt = f"""## 社内資料
{context}

## 質問
{question.text}

上記の社内資料の各セクションを参照して回答してください。
回答には必ず参照したセクション（例: attendance.md - 有給休暇）を明記してください。"""

    # ステップ5: Gemini APIで回答生成
    model = genai.GenerativeModel('models/gemini-2.5-flash-preview-09-2025')  # Gemini 2.5 Flash（プレビュー版）

    # システムプロンプトとユーザープロンプトを結合
    full_prompt = f"""{system_prompt}

{user_prompt}"""

    response = model.generate_content(
        full_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,  # 低めに設定して正確性を重視（説明可能性）
        )
    )

    answer = response.text

    # ステップ6: レスポンスを構築
    # 後方互換性: sources はファイル名のリスト（重複除去）
    sources = list(set(meta["filename"] for meta in results["metadatas"][0]))

    # sections は詳細なセクション情報（v2.0.0で追加）
    sections = [
        {
            "filename": meta["filename"],
            "section_title": meta["section_title"],
            "similarity_score": round(1.0 - dist, 3)
        }
        for meta, dist in zip(results["metadatas"][0], results["distances"][0])
    ]

    # レスポンス（後方互換性を保ちつつ新機能を追加）
    return {
        "question": question.text,
        "expanded_query": expansion_result.expanded_query,  # v2.1.0で追加
        "added_keywords": expansion_result.added_keywords,  # v2.1.0で追加
        "answer": answer,
        "sources": sources,           # 既存フィールド（後方互換性）
        "sections": sections,          # v2.0.0で追加
        "confidence": confidence,      # v2.0.0で追加
        "version": "2.1.0"            # バージョン更新
    }
