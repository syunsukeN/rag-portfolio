# ============================================
# RAGポートフォリオ Makefile
# ============================================
# 複合コマンドで複数の操作を1コマンドで実行

# デフォルトのシェルを指定
SHELL := /bin/bash

# .PHONYはファイル名と競合しないようにするおまじない
.PHONY: help init fresh rebuild open test status shell nuke

# ============================================
# ヘルプ（make または make help で表示）
# ============================================
help:
	@echo "======================================"
	@echo "  RAGポートフォリオ コマンド一覧"
	@echo "======================================"
	@echo ""
	@echo "【セットアップ・再構築】"
	@echo "  make init     - 初回セットアップ（clone直後に実行）"
	@echo "  make fresh    - 全部やり直し（停止→削除→再ビルド→DB初期化→起動→ブラウザ）"
	@echo "  make rebuild  - DB更新して再起動（DB初期化→再起動→ブラウザ）"
	@echo ""
	@echo "【デバッグ・確認】"
	@echo "  make test     - APIにサンプル質問を投げて動作確認"
	@echo "  make status   - DBの状態確認（チャンク数、ドキュメント一覧）"
	@echo "  make shell    - コンテナ内シェルに入る"
	@echo "  make open     - フロントエンドをブラウザで開く"
	@echo ""
	@echo "【クリーンアップ】"
	@echo "  make nuke     - Docker全削除（イメージ・ボリューム含む完全クリア）"
	@echo ""

# ============================================
# セットアップ・再構築
# ============================================

# 初回セットアップ（clone直後に1回だけ実行）
init:
	@echo "============================================"
	@echo "🎉 初回セットアップを開始します"
	@echo "============================================"
	@echo ""
	@echo "📍 Step 1/4: .envファイルを作成..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "   .env.example を .env にコピーしました"; \
		echo "   ⚠️  .env を編集して CHROMA_GOOGLE_GENAI_API_KEY を設定してください"; \
	else \
		echo "   .env は既に存在します（スキップ）"; \
	fi
	@echo ""
	@echo "📍 Step 2/4: Dockerイメージをビルド..."
	docker compose build
	@echo ""
	@echo "📍 Step 3/4: コンテナを起動..."
	docker compose up -d
	@echo ""
	@echo "📍 Step 4/4: ベクトルDBを初期化..."
	@echo "   (コンテナが起動するまで5秒待機)"
	sleep 5
	docker compose exec -e SKIP_CONFIRMATION=true api python setup_db.py
	@echo ""
	@echo "============================================"
	@echo "✅ 初回セットアップが完了しました！"
	@echo "   API: http://localhost:8000"
	@echo "============================================"
	@echo ""
	@echo "🌐 ブラウザでフロントエンドを開きます..."
	open frontend/index.html
	@echo ""
	@echo "📋 ログを表示します（Ctrl+C で終了）"
	docker compose logs -f api

# 全部やり直し（完全クリーンな状態から再構築）
fresh:
	@echo "============================================"
	@echo "🔄 完全再構築を開始します"
	@echo "============================================"
	@echo ""
	@echo "📍 Step 1/5: Dockerコンテナを停止..."
	-docker compose down
	@echo ""
	@echo "📍 Step 2/5: キャッシュを削除..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo ""
	@echo "📍 Step 3/5: Dockerイメージを再ビルド..."
	docker compose build
	@echo ""
	@echo "📍 Step 4/5: コンテナを起動..."
	docker compose up -d
	@echo ""
	@echo "📍 Step 5/5: ベクトルDBを初期化..."
	@echo "   (コンテナが起動するまで5秒待機)"
	sleep 5
	docker compose exec -e SKIP_CONFIRMATION=true api python setup_db.py
	@echo ""
	@echo "============================================"
	@echo "✅ 完全再構築が完了しました！"
	@echo "   API: http://localhost:8000"
	@echo "============================================"
	@echo ""
	@echo "🌐 ブラウザでフロントエンドを開きます..."
	open frontend/index.html
	@echo ""
	@echo "📋 ログを表示します（Ctrl+C で終了）"
	docker compose logs -f api

# DB更新して再起動（Dockerは停止せず）
rebuild:
	@echo "============================================"
	@echo "🔄 DB更新 & 再起動を開始します"
	@echo "============================================"
	@echo ""
	@echo "📍 Step 1/3: ベクトルDBを初期化..."
	docker compose exec -e SKIP_CONFIRMATION=true api python setup_db.py
	@echo ""
	@echo "📍 Step 2/3: コンテナを再起動..."
	docker compose restart
	@echo ""
	@echo "📍 Step 3/3: 起動を待機..."
	sleep 3
	@echo ""
	@echo "============================================"
	@echo "✅ DB更新 & 再起動が完了しました！"
	@echo "   API: http://localhost:8000"
	@echo "============================================"
	@echo ""
	@echo "🌐 ブラウザでフロントエンドを開きます..."
	open frontend/index.html
	@echo ""
	@echo "📋 ログを表示します（Ctrl+C で終了）"
	docker compose logs -f api

# ============================================
# デバッグ・確認
# ============================================

# APIにサンプル質問を投げて動作確認
test:
	@echo "============================================"
	@echo "🧪 API動作テストを実行します"
	@echo "============================================"
	@echo ""
	@echo "📍 Test 1: ヘルスチェック..."
	@curl -s http://localhost:8000/health | python3 -m json.tool || echo "❌ APIに接続できません"
	@echo ""
	@echo "📍 Test 2: サンプル質問（有給休暇について）..."
	@curl -s -X POST http://localhost:8000/ask \
		-H "Content-Type: application/json" \
		-d '{"text": "有給休暇の申請方法を教えてください"}' | python3 -m json.tool || echo "❌ 質問APIでエラー"
	@echo ""
	@echo "📍 Test 3: 検索API..."
	@curl -s -X POST http://localhost:8000/search \
		-H "Content-Type: application/json" \
		-d '{"text": "経費精算"}' | python3 -m json.tool || echo "❌ 検索APIでエラー"
	@echo ""
	@echo "============================================"
	@echo "✅ テスト完了"
	@echo "============================================"

# DBの状態確認（チャンク数、ドキュメント一覧）
status:
	@echo "============================================"
	@echo "📊 DB状態を確認します"
	@echo "============================================"
	@echo ""
	docker compose exec api python3 -c "\
import chromadb; \
client = chromadb.PersistentClient(path='./chroma_db'); \
col = client.get_collection('company_docs'); \
data = col.get(); \
print(f'📁 総チャンク数: {len(data[\"ids\"])}'); \
print(''); \
print('📄 ドキュメント一覧:'); \
sources = set(); \
for m in data['metadatas']: \
    sources.add(m.get('source', 'unknown')); \
for s in sorted(sources): \
    count = sum(1 for m in data['metadatas'] if m.get('source') == s); \
    print(f'   - {s} ({count} chunks)'); \
"
	@echo ""
	@echo "============================================"

# コンテナ内シェルに入る
shell:
	@echo "🐚 コンテナ内シェルに入ります（exit で終了）"
	docker compose exec api /bin/bash

# フロントエンドをブラウザで開く
open:
	@echo "🌐 ブラウザでフロントエンドを開きます..."
	open frontend/index.html

# ============================================
# クリーンアップ
# ============================================

# Docker全削除（イメージ・ボリューム含む完全クリア）
nuke:
	@echo "============================================"
	@echo "💥 Docker環境を完全削除します"
	@echo "============================================"
	@echo ""
	@echo "⚠️  この操作は以下を削除します:"
	@echo "   - コンテナ"
	@echo "   - イメージ"
	@echo "   - ボリューム（DBデータ含む）"
	@echo ""
	@read -p "本当に実行しますか？ (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	@echo ""
	@echo "📍 Step 1/2: コンテナとボリュームを削除..."
	docker compose down -v --rmi local
	@echo ""
	@echo "📍 Step 2/2: キャッシュを削除..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo ""
	@echo "============================================"
	@echo "✅ 完全削除が完了しました"
	@echo "   再構築するには: make init"
	@echo "============================================"
