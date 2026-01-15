"""
クエリ拡張モジュール - ベクトル検索の精度向上のためのキーワード拡張

このモジュールは、ユーザーの質問文をLLMで解析し、
ドメイン固有のキーワードを追加することで検索精度を向上させます。

## 背景と問題
ベクトル検索では「申請」「方法」などの一般語が類似度を支配してしまい、
ドメイン（勤怠/経費/IT）の違いが無視される問題がありました。

例:
- 「有給の申請方法は？」→ IT系の「ソフトウェア申請」がヒット（本来は勤怠系）
- 「経費精算の締め切りは？」→ 勤怠系の「申請期限」がヒット（本来は経費系）

## 解決策: クエリ拡張
LLMで質問を分析し、ドメイン固有のキーワードを追加してから検索します。
「有給の申請方法」→「有給の申請方法 有給休暇 勤怠 休み」

## 技術選定理由
- Gemini API: 既存のLLMインフラを再利用（追加コスト最小化）
- シンプルなプロンプト設計: 再現性と説明可能性を重視
- フォールバック機構: API障害時も元のクエリで検索を継続（堅牢性）

## 使用例
```python
expander = QueryExpander()
result = expander.expand("有給の申請方法は？")
print(result.expanded_query)  # "有給の申請方法は？ 有給休暇 勤怠 休み 勤怠くん"
print(result.added_keywords)  # ["有給休暇", "勤怠", "休み", "勤怠くん"]
```
"""

import os
import json
import re
import logging
from dataclasses import dataclass
from typing import Optional, List
import google.generativeai as genai
from dotenv import load_dotenv

# 環境変数を読み込む
load_dotenv()

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExpansionResult:
    """
    クエリ拡張の結果を格納するデータクラス

    この結果オブジェクトには、クエリ拡張の成功/失敗に関わらず
    検索に使用できるクエリが含まれます（堅牢性の確保）。

    Attributes:
        original_query (str): 元のユーザークエリ
        expanded_query (str): 拡張後のクエリ（検索に使用）
            - 成功時: 元のクエリ + 追加キーワード
            - 失敗時: 元のクエリのまま（フォールバック）
        added_keywords (List[str]): LLMが追加したキーワードのリスト
            - 成功時: ["有給休暇", "勤怠", "休み"] など
            - 失敗時: [] 空リスト
        expansion_applied (bool): 拡張が成功したかどうか
            - True: LLMによるキーワード追加が成功
            - False: 無効化、API障害、パースエラーなど
        error_message (Optional[str]): エラー発生時のメッセージ
            - 成功時: None
            - 失敗時: エラーの詳細（ログ・デバッグ用）

    使用例:
        result = ExpansionResult(
            original_query="有給の申請方法は？",
            expanded_query="有給の申請方法は？ 有給休暇 勤怠",
            added_keywords=["有給休暇", "勤怠"],
            expansion_applied=True
        )
    """
    original_query: str
    expanded_query: str
    added_keywords: List[str]
    expansion_applied: bool
    error_message: Optional[str] = None


class QueryExpander:
    """
    LLMを使用してクエリを拡張するクラス

    ベクトル検索の精度を向上させるため、ユーザーの質問文を
    LLM（Gemini）で分析し、ドメイン固有のキーワードを追加します。

    設計思想:
        - 再現性: temperature=0.1 で決定論的な出力
        - 堅牢性: API障害時は元のクエリで継続
        - 説明可能性: 追加したキーワードをレスポンスに含める
        - 変更容易性: 環境変数で有効/無効を切り替え可能

    Attributes:
        model_name (str): 使用するGeminiモデル名
        temperature (float): 生成の多様性（低いほど決定論的）
        max_keywords (int): 追加するキーワードの最大数
        enabled (bool): クエリ拡張を有効にするか

    使用例:
        # 基本的な使い方
        expander = QueryExpander()
        result = expander.expand("有給の申請方法は？")

        # 環境変数で無効化（テスト・デバッグ用）
        # export QUERY_EXPANSION_ENABLED=false
        expander = QueryExpander(enabled=False)
    """

    # クエリ拡張用のプロンプトテンプレート
    # 社内資料のドメイン知識を組み込み、適切なキーワード追加を促す
    EXPANSION_PROMPT_TEMPLATE = """あなたは社内資料検索システムの検索アシスタントです。
ユーザーの質問を分析し、ベクトル検索の精度を向上させるための追加キーワードを提案してください。

## ドメイン知識
社内資料には以下のカテゴリがあります:
- 勤怠管理（勤務時間、有給休暇、遅刻・早退、フレックス）→ キーワード例: 勤怠, 有給休暇, 休み, 勤怠くん, attendance
- 経費精算（交通費、宿泊費、接待、書籍、領収書）→ キーワード例: 経費, 精算, 経費くん, expense, 領収書
- ITサポート（パスワード、PC、Wi-Fi、ソフトウェア、インストール）→ キーワード例: IT, サポート, ヘルプデスク, インストール, ソフトウェア

## タスク
ユーザーの質問に対して、検索精度を向上させる関連キーワードを最大{max_keywords}個提案してください。

## ルール
1. 質問の意図を正確に理解し、関連するカテゴリを特定する
2. そのカテゴリに固有のキーワードを追加する
3. 一般的すぎるキーワード（「方法」「申請」「について」など）は追加しない
4. **JSON配列のみを出力**。例: ["有給休暇", "勤怠", "休み"]
5. 説明文・注釈は一切禁止
6. 各キーワードは15文字以内の単語または短いフレーズ

## 質問
{query}

## 追加キーワード（JSON配列のみ）:"""

    def __init__(
        self,
        model_name: str = "models/gemini-2.5-flash-preview-09-2025",
        temperature: float = 0.1,
        max_keywords: int = 5,
        enabled: bool = True
    ):
        """
        QueryExpanderを初期化

        Args:
            model_name (str): 使用するGeminiモデル名
                デフォルト: gemini-2.5-flash（高速・低コスト）
            temperature (float): 生成の多様性（0.0-1.0）
                低いほど決定論的で再現性が高い
                デフォルト: 0.1（ほぼ決定論的）
            max_keywords (int): 追加するキーワードの最大数
                多すぎると検索がノイズまみれになる
                デフォルト: 5
            enabled (bool): クエリ拡張を有効にするか
                環境変数 QUERY_EXPANSION_ENABLED で制御可能
                デフォルト: True

        Note:
            APIキーは環境変数 CHROMA_GOOGLE_GENAI_API_KEY から自動で読み込まれます。
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_keywords = max_keywords
        self.enabled = enabled

        # Gemini APIの設定（main.pyと同じAPIキーを使用）
        api_key = os.getenv("CHROMA_GOOGLE_GENAI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)

        logger.info(f"QueryExpander initialized: enabled={enabled}, model={model_name}")

    def expand(self, query: str) -> ExpansionResult:
        """
        ユーザークエリを拡張する

        LLMを使用して質問を分析し、ドメイン固有のキーワードを追加します。
        エラー発生時も検索が継続できるよう、元のクエリをフォールバックとして返します。

        Args:
            query (str): ユーザーの元のクエリ

        Returns:
            ExpansionResult: 拡張結果
                - 成功時: expanded_query に拡張後のクエリ
                - 失敗時: expanded_query に元のクエリ（フォールバック）

        Raises:
            ValueError: クエリが空の場合（停止可能性の確保）

        処理フロー:
            1. 入力バリデーション（空クエリチェック）
            2. 機能が無効の場合は早期リターン
            3. 拡張プロンプトの構築
            4. Gemini APIの呼び出し
            5. レスポンスのパース
            6. 元のクエリ + キーワードの結合

        使用例:
            expander = QueryExpander()
            result = expander.expand("有給の申請方法は？")

            if result.expansion_applied:
                print(f"拡張成功: {result.expanded_query}")
            else:
                print(f"拡張スキップ: {result.error_message}")
        """
        # 入力バリデーション（停止可能性 - 不正な入力で明示的に停止）
        if not query or not query.strip():
            raise ValueError("クエリが空です")

        query = query.strip()

        # 機能が無効の場合は元のクエリをそのまま返却
        if not self.enabled:
            logger.info("Query expansion is disabled")
            return ExpansionResult(
                original_query=query,
                expanded_query=query,
                added_keywords=[],
                expansion_applied=False,
                error_message="Query expansion is disabled"
            )

        try:
            # プロンプトを構築
            prompt = self._build_prompt(query)

            # Gemini APIを呼び出し
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=100  # キーワードのみなので短く
                )
            )

            # レスポンスからキーワードを抽出
            keywords = self._parse_keywords(response.text)

            # キーワードが取得できなかった場合
            if not keywords:
                logger.warning(f"No keywords extracted for query: {query}")
                return ExpansionResult(
                    original_query=query,
                    expanded_query=query,
                    added_keywords=[],
                    expansion_applied=False,
                    error_message="No keywords extracted from LLM response"
                )

            # 元のクエリ + キーワードを結合して拡張クエリを作成
            expanded_query = f"{query} {' '.join(keywords)}"

            logger.info(f"Query expanded: '{query}' -> '{expanded_query}'")

            return ExpansionResult(
                original_query=query,
                expanded_query=expanded_query,
                added_keywords=keywords,
                expansion_applied=True
            )

        except Exception as e:
            # APIエラー時は元のクエリで継続（堅牢性）
            logger.warning(f"Query expansion failed, using original query: {e}")

            return ExpansionResult(
                original_query=query,
                expanded_query=query,  # フォールバック
                added_keywords=[],
                expansion_applied=False,
                error_message=str(e)
            )

    def _build_prompt(self, query: str) -> str:
        """
        クエリ拡張用のプロンプトを構築

        社内資料のドメイン知識（勤怠/経費/IT）を含むプロンプトを生成し、
        LLMが適切なキーワードを提案できるようにします。

        Args:
            query (str): ユーザーの質問

        Returns:
            str: LLMに送信するプロンプト
        """
        return self.EXPANSION_PROMPT_TEMPLATE.format(
            max_keywords=self.max_keywords,
            query=query
        )

    def _parse_keywords(self, response_text: str) -> List[str]:
        """
        LLMのレスポンスからキーワードを抽出

        パース戦略:
            1. JSON配列の抽出を試みる
            2. 失敗時は従来のカンマ方式にフォールバック
            3. 最小ガード（箇条書き除去、括弧注釈除去、長さ制限）

        Args:
            response_text (str): LLMからのレスポンステキスト

        Returns:
            List[str]: 抽出したキーワードのリスト（最大max_keywords個）
        """
        if not response_text:
            return []

        text = response_text.strip()
        candidates = []

        # 1) JSON配列の抽出を試みる（非貪欲で最短マッチ）
        # 貪欲だと ["a","b"]\n補足: [注意] のような場合に最後の]まで取ってしまう
        json_match = re.search(r"\[[\s\S]*?\]", text)
        if json_match:
            try:
                arr = json.loads(json_match.group(0))
                if isinstance(arr, list):
                    candidates = [str(x).strip() for x in arr]
            except json.JSONDecodeError:
                logger.debug(f"[PARSE] JSON parse failed, falling back to comma split")

        # 2) フォールバック（従来方式）
        if not candidates:
            raw = text.replace("\n", ",")
            raw = raw.replace("、", ",").replace("・", ",")
            candidates = [x.strip() for x in raw.split(",") if x.strip()]

        # 3) 最小ガード
        cleaned = []
        rejected_count = 0

        for kw in candidates:
            original_kw = kw

            # 箇条書き記号除去（-, *, 1. など）
            kw = re.sub(r"^[\-\*\d\.\)\]]+\s*", "", kw)

            # クォート・角括弧除去（JSON失敗時のフォールバックで残るゴミを除去）
            kw = re.sub(r"[\"'`\[\]]", "", kw)

            # 末尾の括弧注釈除去（「勤怠（カテゴリです）」→「勤怠」）
            kw = re.sub(r"[（(][^）)]*[）)]$", "", kw).strip()

            if not kw:
                rejected_count += 1
                logger.debug(f"[GUARD] Rejected (empty after clean): '{original_kw}'")
                continue

            # 長さ制限（24文字）
            if len(kw) > 24:
                rejected_count += 1
                logger.debug(f"[GUARD] Rejected (too long): '{kw}'")
                continue

            cleaned.append(kw)

        # 集計ログ（info）
        if rejected_count > 0:
            logger.info(f"[GUARD] rejected={rejected_count} kept={len(cleaned)}")

        logger.debug(f"Parsed keywords: {cleaned}")
        return cleaned[:self.max_keywords]


# 単体テスト用のメイン処理
if __name__ == "__main__":
    """
    単体テスト用のメイン処理

    実行方法:
        python3 query_expander.py

    期待される出力:
        各テストクエリに対する拡張結果
    """
    print("=" * 60)
    print("クエリ拡張モジュール - 単体テスト")
    print("=" * 60)
    print()

    # テスト用クエリ（元の問題ケース）
    test_queries = [
        "有給の申請方法は？",
        "経費精算の締め切りは？",
        "業務用ソフトのインストール方法"
    ]

    expander = QueryExpander()

    for query in test_queries:
        print(f"元のクエリ: {query}")
        result = expander.expand(query)
        print(f"拡張クエリ: {result.expanded_query}")
        print(f"追加キーワード: {result.added_keywords}")
        print(f"拡張適用: {result.expansion_applied}")
        if result.error_message:
            print(f"エラー: {result.error_message}")
        print("-" * 40)

    print()
    print("=" * 60)
    print("テスト完了")
    print("=" * 60)
