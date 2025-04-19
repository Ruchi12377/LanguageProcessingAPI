"""
言語処理APIのメインエントリーポイント
"""
import os
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

from app import create_app

# アプリケーションの作成
app = create_app()

if __name__ == "__main__":
    # 環境変数からポート番号を取得（デフォルト：8080）
    port = int(os.environ.get("PORT", 8080))
    # デバッグモードの設定（本番環境では無効にすべき）
    debug = os.environ.get("FLASK_ENV") == "development"
    
    # アプリケーションの実行
    app.run(debug=debug, host="0.0.0.0", port=port)
