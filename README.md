# 社内資料検索・質問応答AI（RAGベース）

業務特化型のRAG（Retrieval-Augmented Generation）システムのポートフォリオです。

## 🎯 このプロジェクトで示したいこと

| 観点 | 実装内容 |
|------|----------|
| AIを直接信用しない設計 | プロンプトで「資料にないことは答えない」よう制御 |
| 業務データに限定した回答 | ベクトルDBに登録した文書のみを参照 |
| 参照元の明示 | 回答と共に根拠となる資料名を返却 |
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
├── main.py            # FastAPIサーバー
├── setup_db.py        # ドキュメントのベクトル化
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

### レスポンス例

```json
{
  "question": "有給休暇は何日もらえますか？",
  "answer": "入社6ヶ月後に10日付与されます。",
  "sources": ["attendance.md"]
}
```

## 🛡️ 設計上の工夫

### 1. ハルシネーション対策

```python
system_prompt = """
1. 提供された社内資料のみを参照して回答してください
2. 資料に書かれていない情報は「その情報は社内資料に記載がありません」と回答してください
"""
```

### 2. 回答の根拠を明示

検索にヒットした資料名を`sources`として返却し、回答の信頼性を担保。

### 3. 検索精度の確保

Gemini Embedding（text-embedding-004）を使用し、意味的な類似検索を実現。

## 📝 今後の改善点

- [ ] ドキュメントのチャンク分割で検索精度向上
- [ ] 検索スコアの閾値設定
- [ ] ユーザー認証機能
- [ ] 回答のストリーミング対応
- [ ] Docker化

## 📄 ライセンス

MIT License
