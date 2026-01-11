# 社内資料検索・質問応答AI（RAGベース）

業務特化型のRAG（Retrieval-Augmented Generation）システムのポートフォリオです。

**現在のバージョン: v2.1.0**（チャンク分割 + クエリ拡張対応）

## 🎯 このプロジェクトで示したいこと

| 観点 | 実装内容 |
|------|----------|
| AIを直接信用しない設計 | プロンプトで「資料にないことは答えない」よう制御 |
| 業務データに限定した回答 | ベクトルDBに登録した文書のみを参照 |
| 参照元の明示 | 回答と共に根拠となる資料名・セクション名を返却 |
| 検索精度の向上 | チャンク分割・クエリ拡張・見出し込み埋め込み |
| 信頼度の可視化 | 検索結果の信頼度レベル（high/medium/low）を表示 |
| 実運用を想定したAPI設計 | FastAPIによるRESTful API |

## 🏗️ アーキテクチャ

```text
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Frontend   │────▶│   FastAPI   │────▶│   Gemini    │
│  (HTML/JS)  │◀────│   Backend   │◀────│   API       │
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  ChromaDB   │
                    │ (VectorDB)  │
                    └─────────────┘
```

## 🔧 技術スタック

| 項目 | 技術 |
|------|------|
| バックエンド | Python / FastAPI |
| ベクトルDB | ChromaDB |
| Embedding | Gemini text-embedding-004 |
| LLM | Gemini 2.5 Flash |
| フロントエンド | HTML / CSS / JavaScript |

## 📁 ディレクトリ構成

```text
rag-portfolio/
├── main.py            # FastAPIサーバー（v2.1.0）
├── setup_db.py        # ドキュメントのベクトル化
├── chunker.py         # チャンク分割モジュール（H2見出し単位）
├── query_expander.py  # クエリ拡張モジュール（LLMキーワード追加）
├── requirements.txt   # 依存ライブラリ
├── .env.example       # 環境変数テンプレート
├── data/
│   └── docs/          # 社内資料（サンプル）
└── frontend/
    └── index.html     # 質問UI
```

## 🚀 セットアップ

### 1. リポジトリをクローン

```bash
git clone https://github.com/yourusername/rag-portfolio.git
cd rag-portfolio
```

### 2. 仮想環境を作成・有効化

```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows
```

### 3. ライブラリをインストール

```bash
pip3 install -r requirements.txt
```

### 4. 環境変数を設定

```bash
cp .env.example .env
# .env を編集してAPIキーを設定
```

### 5. ドキュメントをベクトル化

```bash
python3 setup_db.py
```

### 6. サーバーを起動

```bash
python3 -m uvicorn main:app --reload
```

### 7. フロントエンドにアクセス

`frontend/index.html` をブラウザで開く

## 📡 API エンドポイント

| メソッド | パス | 説明 |
|---------|------|------|
| GET | `/` | ルート（動作確認） |
| GET | `/health` | ヘルスチェック |
| POST | `/search` | ドキュメント検索 |
| POST | `/ask` | 質問応答（RAG） |

### `/ask` リクエスト例

```json
{
  "text": "有給休暇は何日もらえますか？"
}
```

### レスポンス例（v2.1.0）

```json
{
  "question": "有給休暇は何日もらえますか？",
  "expanded_query": "有給休暇は何日もらえますか？ 有給 勤怠 休み",
  "added_keywords": ["有給", "勤怠", "休み"],
  "answer": "入社6ヶ月後に10日付与されます。",
  "sources": ["attendance.md"],
  "sections": [
    {
      "filename": "attendance.md",
      "section_title": "有給休暇",
      "similarity_score": 0.85
    }
  ],
  "confidence": "high",
  "version": "2.1.0"
}
```

| フィールド | 説明 |
|------------|------|
| expanded_query | クエリ拡張後の検索クエリ |
| added_keywords | LLMが追加したキーワード |
| sections | 参照したセクションの詳細情報 |
| confidence | 検索結果の信頼度（high/medium/low） |
| version | APIバージョン |

## 🛡️ 設計上の工夫

### 1. ハルシネーション対策

```python
system_prompt = """
1. 提供された社内資料のみを参照して回答してください
2. 資料に書かれていない情報は「その情報は社内資料に記載がありません」と回答してください
"""
```

### 2. 回答の根拠を明示

検索にヒットした資料名・セクション名を`sources`と`sections`で返却し、回答の信頼性を担保。

### 3. チャンク分割（v2.0.0で追加）

ドキュメント全体ではなく、H2見出し単位でセクションに分割して検索。

```
Before: attendance.md 全体（4セクション分）→ AIに渡す
After:  有給休暇セクションのみ → AIに渡す（情報量60%削減）
```

**メリット:**
- 必要な情報だけをAIに渡せる
- レイテンシとコストの削減
- どのセクションを参照したか追跡可能

### 4. 信頼度スコア（v2.0.0で追加）

検索結果の類似度に基づいて信頼度レベルを計算。

| 信頼度 | 類似度 | 意味 |
|--------|--------|------|
| high | 0.8以上 | ほぼ完全一致、自信を持って回答 |
| medium | 0.6以上 | 関連性あり、注意付きで回答 |
| low | 0.6未満 | 関連性薄い、回答を控える |

### 5. クエリ拡張（v2.1.0で追加）

LLMでドメイン固有のキーワードを自動追加し、検索精度を向上。

```
Before: 「有給の申請方法は？」
After:  「有給の申請方法は？ 有給休暇 勤怠 休み 勤怠くん」
```

**フォールバック設計:** API障害時も元のクエリで検索を継続。

### 6. 見出し込み埋め込み（v2.1.0で追加）

H2見出しをチャンクの先頭に含めることで、ドメイン識別を強化。

```
Before: 「- 有給休暇は入社6ヶ月後に10日付与されます」
After:  「## 有給休暇\n- 有給休暇は入社6ヶ月後に10日付与されます」
```

## 📝 今後の改善点

### 完了済み
- [x] ドキュメントのチャンク分割で検索精度向上（v2.0.0）
- [x] 検索スコアの閾値設定・信頼度表示（v2.0.0）
- [x] クエリ拡張機能（v2.1.0）
- [x] 見出し込み埋め込み（v2.1.0）

### 次のフェーズ
- [ ] ハイブリッド検索（ベクトル + キーワード検索の組み合わせ）
- [ ] リランキング（検索後にLLMで並べ替え）
- [ ] 評価用質問セットとRecall@k計測
- [ ] 抽象的なクエリへの対応（聞き返しUI）

### 将来的な拡張
- [ ] ユーザー認証機能
- [ ] 回答のストリーミング対応
- [ ] Docker化

## 📄 ライセンス

MIT License
