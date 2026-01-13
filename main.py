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
import logging
from dataclasses import dataclass
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# ロギング設定（聞き返し機能のデバッグ用）
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    version="2.2.1"
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


# ============================================================
# グローバル例外ハンドラ（v2.2.1で追加）
# Gemini APIのクォータ制限エラーをキャッチして適切なJSONを返す
# ============================================================
@app.exception_handler(google_exceptions.ResourceExhausted)
async def quota_exceeded_handler(request, exc):
    """
    Gemini APIのクォータ制限エラーをハンドリング

    フリープランでは1日20リクエストの制限があり、
    超過するとResourceExhausted例外が発生する。
    フロントエンドで分かりやすく表示するために、
    error_typeを含むJSONレスポンスを返す。
    """
    logger.warning(f"[QUOTA] API quota exceeded: {exc}")
    return JSONResponse(
        status_code=429,  # Too Many Requests
        content={
            "error_type": "quota_exceeded",
            "message": "APIの利用制限に達しました。フリープランでは1日20リクエストまでです。",
            "detail": "しばらく時間をおいてから再度お試しください（翌日にリセットされます）。",
        }
    )

# Google Gemini APIの設定
# 埋め込み（Embedding）とLLM応答の両方で使用
genai.configure(api_key=os.getenv("CHROMA_GOOGLE_GENAI_API_KEY"))

# ChromaDBの設定
# PersistentClient: ディスクに永続化（./chroma_db/）
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Gemini Embedding関数を設定
# gemini-embedding-001: 3072次元ベクトルを生成（日本語対応）
# 注意: text-embedding-004 は日本語で同一ベクトルを返すバグがあるため移行
gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    model_name="models/gemini-embedding-001"
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
        filter_file (Optional[str]): 検索対象を絞り込むファイル名（v2.2.0で追加）
            聞き返し機能の選択肢クリック時に使用。
            例: "expense.md" → 経費精算関連のドキュメントのみを検索
        chunk_id (Optional[str]): 直接取得するチャンクID（v2.2.0 Solution Cで追加）
            聞き返し選択肢クリック時に使用。
            これが指定されている場合、検索をスキップしてチャンクを直接取得する。
            → 無限ループの構造的な防止（検索しない = 曖昧判定が走らない）
    """
    text: str
    filter_file: Optional[str] = None  # ハードフィルタ用（v2.2.0で追加）
    chunk_id: Optional[str] = None     # 直接取得用（v2.2.0 Solution Cで追加）


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


# ============================================================
# 聞き返し機能（Clarification）v2.2.0で追加
# ============================================================

# 聞き返し判定の閾値設定
# 注意: Solution B（ハードフィルタ）により、選択肢クリック時はドメインで絞り込み検索を行う
CLARIFICATION_THRESHOLDS = {
    "min_similarity": 0.8,    # 条件①: これ未満で聞き返し
    "ambiguity_gap": 0.03,    # 条件②: 1位-2位の差がこれ未満で聞き返し
}

# ドメインマッピング（ファイル名 → 日本語ドメイン名）
DOMAIN_MAP = {
    "attendance.md": "勤怠管理",
    "expense.md": "経費精算",
    "it_support.md": "ITサポート",
}


@dataclass
class ClarificationResult:
    """
    聞き返し判定の結果を格納するデータクラス

    Attributes:
        should_clarify (bool): 聞き返しが必要かどうか
        reason (str): 判定理由
            - "low_similarity": 類似度が低い
            - "ambiguous_ranking": 1位と2位の差が小さい
            - "scattered_domains": ドメインが分散している
            - "no_results": 検索結果なし
            - "none": 聞き返し不要
        message (str): ユーザーに表示するメッセージ
        options (List[dict]): 選択肢 [{"label": "有給休暇", "query": "..."}]
    """
    should_clarify: bool
    reason: str
    message: str
    options: List[dict]


def _filename_to_domain(filename: str) -> str:
    """ファイル名から日本語ドメイン名に変換"""
    return DOMAIN_MAP.get(filename, filename.replace(".md", ""))


def _extract_domains(metadatas: list) -> List[str]:
    """メタデータからドメイン名を抽出"""
    return [_filename_to_domain(m["filename"]) for m in metadatas]


def _extract_section_options(metadatas: list, ids: list, original_query: str) -> List[dict]:
    """
    セクションタイトルからユーザーに提示する選択肢を生成

    Solution C（chunk_id方式）: 各選択肢に chunk_id を含める
    → クリック時に検索をスキップしてチャンクを直接取得
    → 無限ループの構造的な防止（検索しない = 曖昧判定が走らない）

    Args:
        metadatas: results["metadatas"][0] から渡す
        ids: results["ids"][0] から渡す（chunk_id用）
        original_query: 元のクエリ（現在は未使用だが将来の拡張用）

    Returns:
        [{"label": "有給休暇（勤怠管理）", "query": "...", "filter_file": "attendance.md", "chunk_id": "..."}]

    設計メモ:
        - zip(ids, metadatas) を使用してインデックスのミスマッチを防止
        - continue で要素をスキップしても、id と metadata のペアは常に正しい
    """
    options = []
    seen_labels = set()

    # zip で id と metadata を安全にペアリング
    # enumerate だと continue 時にインデックスがずれるリスクがある
    for doc_id, m in zip(ids, metadatas):
        section = m.get("section_title", "")
        # 無効なセクションはスキップ
        if not section or section in ["(Preamble)", "(No Section)"]:
            continue

        domain = _filename_to_domain(m["filename"])
        label = f"{section}（{domain}）"

        # 重複を除外
        if label in seen_labels:
            continue
        seen_labels.add(label)

        options.append({
            "label": label,
            "query": f"{section}について教えてください",
            "filter_file": m["filename"],  # ハードフィルタ用（フォールバック）
            "chunk_id": doc_id              # 直接取得用（Solution C）
        })

        # 最大3件まで
        if len(options) >= 3:
            break

    return options


def should_clarify(results, filter_file: Optional[str] = None) -> ClarificationResult:
    """
    検索結果を分析し、聞き返しが必要かどうかを判定

    判定条件（3つのうち1つでも該当すれば聞き返し）:
        ① 低類似度: top_similarity < 0.8
        ② 曖昧なランキング: (top1 - top2) < 0.03
           ※ filter_file がある場合は基本スキップ（ドメイン絞り込み済み）
           ※ ただし同一ドメイン内で複数セクションが曖昧なら聞き返し
        ③ ドメイン分散: top3が3種類のドメインから

    Args:
        results: ChromaDB query() の戻り値
        filter_file: ハードフィルタ用のファイル名（v2.2.0で追加）
            これが指定されている場合、ドメイン絞り込み済みと判断

    Returns:
        ClarificationResult: 判定結果

    設計原則:
        - この関数は検索結果のみを見て判断する
        - calculate_confidence() とは役割を分離（循環依存を避ける）
    """
    # エッジケース: 結果が空
    if not results["documents"] or not results["documents"][0]:
        logger.info("clarification: reason=no_results")
        # filter_file がある場合は空結果でも聞き返ししない（回答不能として処理）
        if filter_file:
            return ClarificationResult(
                should_clarify=False,
                reason="none",
                message="",
                options=[]
            )
        return ClarificationResult(
            should_clarify=True,
            reason="no_results",
            message="該当する情報が見つかりませんでした。以下のカテゴリから選んでください。",
            options=[
                {"label": "勤怠管理（有給・勤務時間など）", "query": "勤怠管理について教えてください", "filter_file": "attendance.md"},
                {"label": "経費精算（交通費・領収書など）", "query": "経費精算について教えてください", "filter_file": "expense.md"},
                {"label": "ITサポート（パスワード・PCなど）", "query": "ITサポートについて教えてください", "filter_file": "it_support.md"},
            ]
        )

    # 類似度スコアを計算
    distances = results["distances"][0]
    similarities = [1.0 - d for d in distances]
    metadatas = results["metadatas"][0]
    ids = results["ids"][0]  # chunk_id用（Solution C）

    top_similarity = similarities[0]
    similarity_gap = similarities[0] - similarities[1] if len(similarities) > 1 else 1.0
    domains = _extract_domains(metadatas)
    unique_domains = list(set(domains))

    # 条件①: 低類似度チェック
    if top_similarity < CLARIFICATION_THRESHOLDS["min_similarity"]:
        options = _extract_section_options(metadatas, ids, "")
        logger.info(f"clarification: reason=low_similarity, top_sim={top_similarity:.3f}")
        return ClarificationResult(
            should_clarify=True,
            reason="low_similarity",
            message="お探しの情報と完全に一致する資料が見つかりませんでした。\nどのカテゴリに関する質問ですか？",
            options=options if options else [
                {"label": "勤怠管理", "query": "勤怠管理について教えてください", "filter_file": "attendance.md"},
                {"label": "経費精算", "query": "経費精算について教えてください", "filter_file": "expense.md"},
                {"label": "ITサポート", "query": "ITサポートについて教えてください", "filter_file": "it_support.md"},
            ]
        )

    # 条件②: 曖昧なランキングチェック
    if len(similarities) > 1 and similarity_gap < CLARIFICATION_THRESHOLDS["ambiguity_gap"]:
        if filter_file:
            # ハードフィルタ済み: 同一ドメイン内でセクションが曖昧な場合のみ聞き返し
            # 異なるセクションが存在するかチェック
            sections = [m.get("section_title", "") for m in metadatas[:2]]
            if sections[0] != sections[1] and sections[0] not in ["(Preamble)", "(No Section)"]:
                options = _extract_section_options(metadatas[:2], ids[:2], "")
                # filter_file を維持（同一ドメイン内でのさらなる絞り込み用）
                # 注意: chunk_id があるので filter_file の上書きは不要だが、フォールバック用に維持
                for opt in options:
                    opt["filter_file"] = filter_file
                logger.info(f"clarification: reason=ambiguous_within_domain, filter={filter_file}, gap={similarity_gap:.3f}")
                return ClarificationResult(
                    should_clarify=True,
                    reason="ambiguous_within_domain",
                    message=f"{_filename_to_domain(filter_file)}内で複数の項目が見つかりました。\nどちらについてお知りになりたいですか？",
                    options=options
                )
            # 同じセクション or 無効なセクション → 聞き返しスキップ
            logger.info(f"clarification: reason=none (filter applied, same section), filter={filter_file}")
        else:
            # ハードフィルタなし: 従来通り聞き返し
            options = _extract_section_options(metadatas[:2], ids[:2], "")
            logger.info(f"clarification: reason=ambiguous_ranking, gap={similarity_gap:.3f}")
            return ClarificationResult(
                should_clarify=True,
                reason="ambiguous_ranking",
                message="複数の関連情報が見つかりました。どちらについてお知りになりたいですか？",
                options=options
            )

    # 条件③: ドメイン分散チェック（3種類以上のドメインにまたがる）
    # filter_file がある場合はスキップ（すでにドメイン絞り込み済み）
    if not filter_file and len(unique_domains) >= 3:
        logger.info(f"clarification: reason=scattered_domains, domains={unique_domains}")
        # 各ドメインに filter_file を追加
        domain_options = []
        for d in unique_domains[:3]:
            # ドメイン名からファイル名を逆引き
            filename = next((k for k, v in DOMAIN_MAP.items() if v == d), f"{d}.md")
            domain_options.append({
                "label": f"{d}について",
                "query": f"{d}について教えてください",
                "filter_file": filename
            })
        return ClarificationResult(
            should_clarify=True,
            reason="scattered_domains",
            message="複数のカテゴリにまたがる情報が見つかりました。\nどのカテゴリについてお知りになりたいですか？",
            options=domain_options
        )

    # 聞き返し不要
    logger.info(f"clarification: reason=none, top_sim={top_similarity:.3f}, gap={similarity_gap:.3f}, filter={filter_file}")
    return ClarificationResult(
        should_clarify=False,
        reason="none",
        message="",
        options=[]
    )


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
        "message": "RAGポートフォリオ v2.2.1 - 聞き返し機能対応",
        "version": "2.2.1",
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
        "version": "2.2.1"
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
    # ============================================================
    # chunk_id がある場合: 検索せず直接取得（Solution C）
    # これにより無限ループを構造的に防止（検索しない = 曖昧判定が走らない）
    # ============================================================
    if question.chunk_id:
        logger.info(f"[CHUNK] Direct fetch requested: {question.chunk_id}")
        chunk_result = collection.get(ids=[question.chunk_id])

        # chunk_id が見つからない場合のフォールバック
        # DB再構築後など、古いchunk_idが無効になるケースに対応
        if not chunk_result["documents"]:
            logger.warning(f"[CHUNK] Not found: {question.chunk_id}, falling back to search")
            # filter_file があれば絞り込み検索、なければ通常検索
            # ※ この場合でも should_clarify は呼ばれるが、レアケースなので許容
        else:
            # チャンクが見つかった場合: 検索スキップで直接回答生成
            logger.info(f"[CHUNK] Found, generating answer directly")
            meta = chunk_result["metadatas"][0]
            doc = chunk_result["documents"][0]

            # コンテキストを構築
            context = f"【{meta['filename']} - {meta['section_title']}】\n{doc}"

            # LLMで回答生成（should_clarify は呼ばない = 無限ループ防止の核心）
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

            model = genai.GenerativeModel('models/gemini-2.5-flash-preview-09-2025')
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.3)
            )

            return {
                "question": question.text,
                "expanded_query": question.text,  # chunk_id使用時は拡張なし
                "added_keywords": [],
                "response_type": "answer",
                "answer": response.text,
                "clarification": None,
                "sources": [meta["filename"]],
                "sections": [{
                    "filename": meta["filename"],
                    "section_title": meta["section_title"],
                    "similarity_score": 1.0  # 直接取得なので類似度は最高
                }],
                "confidence": "high",  # 直接取得なので信頼度は最高
                "version": "2.2.1"
            }

    # ============================================================
    # 以下は通常のフロー（chunk_id がない or 見つからなかった場合）
    # ============================================================

    # ステップ0: クエリ拡張を適用（v2.1.0で追加）
    # LLMでドメイン固有のキーワードを追加し、検索精度を向上
    # 例: "有給の申請方法" → "有給の申請方法 有給休暇 勤怠 休み"
    expansion_result = query_expander.expand(question.text)

    # ステップ1: 拡張後のクエリで関連チャンクを検索
    # チャンク分割により各セクションが小さくなったため、検索件数を3に設定
    # v2.2.0: filter_file がある場合はハードフィルタを適用（Solution B）
    query_kwargs = {
        "query_texts": [expansion_result.expanded_query],
        "n_results": 3
    }
    if question.filter_file:
        query_kwargs["where"] = {"filename": question.filter_file}
        logger.info(f"[FILTER] Applying hard filter: {question.filter_file}")

    results = collection.query(**query_kwargs)

    # ステップ2: 信頼度レベルを計算
    confidence = calculate_confidence(results)

    # ステップ2.5: 聞き返し判定（v2.2.0で追加）
    # 検索結果が曖昧な場合、LLM呼び出しをスキップして聞き返しレスポンスを返す
    # filter_file がある場合はその情報を渡す（ドメイン絞り込み済みの判定用）
    clarification_result = should_clarify(results, filter_file=question.filter_file)

    if clarification_result.should_clarify:
        # 聞き返しレスポンスを返却（LLM呼び出しなし）
        sources = list(set(meta["filename"] for meta in results["metadatas"][0])) if results["metadatas"][0] else []
        sections = [
            {
                "filename": meta["filename"],
                "section_title": meta["section_title"],
                "similarity_score": round(1.0 - dist, 3)
            }
            for meta, dist in zip(results["metadatas"][0], results["distances"][0])
        ] if results["metadatas"][0] else []

        return {
            "question": question.text,
            "expanded_query": expansion_result.expanded_query,
            "added_keywords": expansion_result.added_keywords,
            "response_type": "clarification",  # v2.2.0で追加
            "answer": None,
            "clarification": {
                "message": clarification_result.message,
                "options": clarification_result.options
            },
            "sources": sources,
            "sections": sections,
            "confidence": confidence,
            "version": "2.2.1"
        }

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
        "response_type": "answer",     # v2.2.0で追加
        "answer": answer,
        "clarification": None,         # v2.2.0で追加（通常回答ではnull）
        "sources": sources,            # 既存フィールド（後方互換性）
        "sections": sections,          # v2.0.0で追加
        "confidence": confidence,      # v2.0.0で追加
        "version": "2.2.1"             # バージョン更新
    }
