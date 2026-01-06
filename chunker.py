"""
Markdownドキュメントをセクション単位でチャンク分割するモジュール

このモジュールは、RAGシステムの検索精度を向上させるために、
Markdownファイルを見出し（H2）ごとに分割します。

技術選定理由:
- 正規表現: 標準ライブラリのみで実現可能、軽量、カスタマイズ容易
- メタデータ駆動設計: ChromaDBのメタデータ機能を活用し、後からフィルタリング可能
- ログ駆動開発: 各処理ステップをログ出力し、問題特定を容易に
"""

import os
import re
from typing import List, Dict, Any
import logging

# ログ設定（INFO レベルで標準出力に出力）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def split_into_chunks(content: str, filename: str) -> List[Dict[str, Any]]:
    """
    Markdownコンテンツをチャンクに分割する

    H1タイトルと最初のH2までを「preamble（序文）」として扱い、
    その後の各H2セクションを個別のチャンクとして分割します。

    Args:
        content (str): Markdownファイルの内容
        filename (str): ファイル名（例: "attendance.md"）

    Returns:
        List[Dict[str, Any]]: チャンクのリスト。各チャンクは以下を含む:
            - id: チャンクID（例: "attendance_0"）
            - content: チャンクの本文
            - metadata: メタデータ（filename, section_title, chunk_index等）

    Raises:
        ValueError: contentが空、またはfilenameが無効な場合

    実装のポイント:
        - H1（# タイトル）+ 序文を最初のチャンクとする
        - 正規表現でH2（## 見出し）を検出して分割
        - 空セクションは警告ログを出力してスキップ
    """
    # 入力チェック（停止可能性 - 不正な入力で異常停止）
    if not content or not content.strip():
        raise ValueError(f"空のコンテンツです: {filename}")

    if not filename or not filename.endswith('.md'):
        raise ValueError(f"無効なファイル名です: {filename}")

    logger.info(f"処理開始: {filename}")

    chunks = []
    filename_base = os.path.splitext(filename)[0]  # "attendance.md" → "attendance"

    # 正規表現パターン（MULTILINE mode）
    # ^ : 行頭
    # ## : H2見出し（H1は#1つ、H3以降は###以上なので除外）
    # \s+ : 空白文字（スペースやタブ）が1文字以上
    # (.+) : 見出しのタイトルをキャプチャ（グループ1）
    # $ : 行末
    h2_pattern = r'^##\s+(.+)$'

    # H2見出しでコンテンツを分割
    # re.split は、パターンにマッチした部分で分割し、キャプチャグループも含めて返す
    # 結果: [preamble, title1, content1, title2, content2, ...]
    parts = re.split(h2_pattern, content, flags=re.MULTILINE)

    # Preamble（H1タイトル + 最初のH2までの内容）の処理
    # 注意: Preambleが短すぎると埋め込みベクトルの品質が低くなり、
    # あらゆるクエリに中途半端にマッチしてしまう問題がある
    preamble = parts[0].strip()
    has_preamble = bool(preamble)
    short_preamble_to_merge = None  # 短いPreambleは最初のH2セクションに結合

    # Preambleが十分な長さ（50文字以上）の場合のみチャンクとして保存
    # 短いPreamble（タイトルのみ等）は検索対象から除外
    MIN_PREAMBLE_LENGTH = 50
    if preamble and len(preamble) >= MIN_PREAMBLE_LENGTH:
        logger.debug(f"Preambleを検出: {len(preamble)} 文字")
        chunks.append({
            "id": f"{filename_base}_0",
            "content": preamble,
            "metadata": {
                "filename": filename,
                "section_title": "(Preamble)",  # 序文を表す特殊な名前
                "chunk_index": 0,
                "chunk_count": 0,  # 後で更新
                "char_count": len(preamble),
                "has_preamble": True
            }
        })
    elif preamble:
        # 短いPreamble: 最初のH2セクションに結合するため保持
        # これによりタイトルが「タグ」として機能し、検索精度が向上する
        short_preamble_to_merge = preamble
        logger.info(f"短いPreambleを最初のセクションに結合予定: {filename} ({len(preamble)}文字)")

    # H2セクションの処理
    # parts[1::2] はタイトル（奇数インデックス）
    # parts[2::2] は本文（偶数インデックス）
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            # タイトルだけで本文がない場合（通常はありえないがエッジケース対応）
            break

        section_title = parts[i].strip()
        section_content = parts[i + 1].strip()

        # 最初のH2セクションに短いPreambleを結合
        # これによりドキュメントタイトル（例: "# 勤怠管理ルール"）が
        # 最初のセクション（例: "## 勤務時間"）と一緒に検索対象になる
        is_first_h2_section = (len(chunks) == 0)
        has_merged_preamble = False
        if is_first_h2_section and short_preamble_to_merge:
            # Preamble + H2見出し + 本文 の形式で結合
            section_content = f"{short_preamble_to_merge}\n\n## {section_title}\n{section_content}"
            has_merged_preamble = True
            logger.info(f"Preambleを最初のセクションに結合: {filename}")

        # 空セクションのチェック（堅牢性 - 空セクションをスキップ）
        if not section_content:
            # 改善可能性: 警告ログで記録し、後で空セクション一覧を確認可能
            logger.warning(f"空セクションをスキップ: {filename} - {section_title}")
            continue

        chunk_idx = len(chunks)  # 現在のチャンク数がインデックスになる
        logger.debug(f"チャンク作成: {section_title} ({len(section_content)} 文字)")

        chunks.append({
            "id": f"{filename_base}_{chunk_idx}",
            "content": section_content,
            "metadata": {
                "filename": filename,
                "section_title": section_title,
                "chunk_index": chunk_idx,
                "chunk_count": 0,  # 後で更新
                "char_count": len(section_content),
                "has_preamble": bool(preamble),  # Preambleの有無
                "has_merged_preamble": has_merged_preamble  # 短いPreambleが結合されたか
            }
        })

    # エッジケース: H2見出しが1つもない場合（堅牢性）
    if not chunks:
        logger.warning(f"H2見出しが見つかりません。ドキュメント全体を1チャンクとして扱います: {filename}")
        chunks.append({
            "id": f"{filename_base}_0",
            "content": content.strip(),
            "metadata": {
                "filename": filename,
                "section_title": "(No Section)",  # セクションなしを表す
                "chunk_index": 0,
                "chunk_count": 1,
                "char_count": len(content.strip()),
                "has_preamble": False
            }
        })

    # 全チャンクの chunk_count を更新（再現性 - 全チャンクに同じ情報）
    total_chunks = len(chunks)
    for chunk in chunks:
        chunk["metadata"]["chunk_count"] = total_chunks

    logger.info(f"{filename} → {total_chunks} chunks に分割完了")
    return chunks


def load_and_chunk_documents(docs_path: str) -> List[Dict[str, Any]]:
    """
    指定ディレクトリ内の全Markdownファイルを読み込み、チャンク分割する

    Args:
        docs_path (str): ドキュメントディレクトリのパス（例: "data/docs"）

    Returns:
        List[Dict[str, Any]]: 全ファイルの全チャンクを含むリスト

    Raises:
        FileNotFoundError: docs_path が存在しない場合
        RuntimeError: .md ファイルが1つも見つからない場合

    エラーハンドリング:
        - ファイル読み込み失敗時はログを記録して継続（堅牢性）
        - 全ファイル処理後に成功したチャンクのみを返却
    """
    # ディレクトリ存在チェック（停止可能性）
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"ディレクトリが見つかりません: {docs_path}")

    # Markdownファイル一覧取得
    all_files = os.listdir(docs_path)
    md_files = [f for f in all_files if f.endswith('.md')]

    if not md_files:
        raise RuntimeError(f".md ファイルが見つかりません: {docs_path}")

    logger.info(f"📂 {len(md_files)} 個のMarkdownファイルを検出")

    all_chunks = []
    failed_files = []  # 失敗したファイルを記録（改善可能性）

    # 各ファイルを処理
    for filename in md_files:
        filepath = os.path.join(docs_path, filename)

        try:
            # ファイル読み込み（UTF-8エンコーディング）
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # チャンク分割
            chunks = split_into_chunks(content, filename)
            all_chunks.extend(chunks)

            # 進捗表示（改善可能性 - どのファイルがどれだけ分割されたか）
            print(f"  📄 {filename} → {len(chunks)} チャンクに分割")

        except Exception as e:
            # エラー発生時はログに記録して継続（堅牢性）
            logger.error(f"ファイル処理失敗: {filename} - {e}")
            failed_files.append(filename)
            continue

    # 処理結果のサマリー
    logger.info(f"合計: {len(all_chunks)} チャンクを作成")

    if failed_files:
        # 失敗したファイルがある場合は警告（改善可能性）
        logger.warning(f"処理失敗: {len(failed_files)} ファイル - {failed_files}")

    # 再現性チェック: チャンク数が0の場合は例外
    if len(all_chunks) == 0:
        raise RuntimeError("チャンクが1つも作成されませんでした")

    return all_chunks


# メイン処理（単体実行用）
if __name__ == "__main__":
    """
    単体テスト用のメイン処理

    実行方法:
        python3 chunker.py

    期待される出力:
        - 各ファイルのチャンク数
        - 合計チャンク数（13個）
        - 各チャンクの詳細情報
    """
    print("=" * 60)
    print("Markdownチャンク分割ツール - 単体テスト")
    print("=" * 60)
    print()

    try:
        # data/docs/ ディレクトリのドキュメントを処理
        chunks = load_and_chunk_documents("data/docs")

        print()
        print("=" * 60)
        print("チャンク一覧")
        print("=" * 60)

        # 各チャンクの詳細を表示
        for chunk in chunks:
            meta = chunk["metadata"]
            print(f"\nID: {chunk['id']}")
            print(f"  ファイル: {meta['filename']}")
            print(f"  セクション: {meta['section_title']}")
            print(f"  文字数: {meta['char_count']}")
            print(f"  インデックス: {meta['chunk_index']} / {meta['chunk_count']}")
            print(f"  Preamble有無: {meta['has_preamble']}")

        print()
        print("=" * 60)
        print(f"✅ テスト成功: 合計 {len(chunks)} チャンク")
        print("=" * 60)

    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ エラー発生: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
