# ============================================
# RAGポートフォリオ Dockerfile
# ============================================
# 目的: 再現可能な実行環境を提供し、どの環境でも同じ動作を保証する
#
# なぜDockerを使うのか:
#   - 再現性の確保: 誰の環境でも同じ動作を保証
#   - 環境構築の簡素化: Python/pipのバージョン差異を吸収
#   - 本番想定の構成: 実運用を見据えたコンテナ化

# ベースイメージ: Python 3.11（安定性とライブラリ互換性を優先）
FROM python:3.11-slim

# 作業ディレクトリを設定
WORKDIR /app

# 依存関係ファイルを先にコピー（Dockerキャッシュ効率化）
COPY requirements.txt .

# Pythonライブラリをインストール
# --no-cache-dir: イメージサイズを削減
# chromadbは最新版でホイール配布されているためbuild-essential不要
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
# data/ はボリュームマウントするためここではコピーしない
COPY main.py chunker.py query_expander.py setup_db.py ./

# フロントエンドもコピー（参照用、実行はホストで行う）
COPY frontend/ ./frontend/

# ポート8000を公開
EXPOSE 8000

# ヘルスチェック（運用時の監視用）
# 30秒ごとに /health エンドポイントを確認
# curlの代わりにpythonを使用（追加パッケージ不要）
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 起動コマンド
# --host 0.0.0.0: コンテナ外からのアクセスを許可
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
