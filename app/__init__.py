"""
アプリケーション初期化モジュール
"""
import os
from flask import Flask, request, abort
import MeCab

# Blueprintのインポート
from app.api.routes.parse import parse_bp
from app.api.routes.similarity import similarity_bp
from app.api.routes.vector import vector_bp
from app.core.plamo_embedding import PlamoEmbedding

def create_app(test_config=None):
    """アプリケーションファクトリー関数
    
    Args:
        test_config: テスト設定（テスト時のみ使用）
        
    Returns:
        Flask: 設定済みのFlaskアプリケーション
    """
    # アプリケーション初期化
    app = Flask(__name__)
    
    # 設定の読み込み
    app.config.from_mapping(
        SECRET_KEY=os.getenv('SECRET_KEY', 'dev'),
        PLAMO_MODEL_NAME=os.getenv('PLAMO_MODEL_NAME', 'pfnet/plamo-embedding-1b'),
        MECAB_DICT_PATH=os.getenv('MECAB_DICT_PATH', '/opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd'),
        API_KEYS=os.getenv('API_KEYS', 'default-key')
    )

    if test_config is None:
        # インスタンス設定があれば、それを読み込む（テストでない場合）
        app.config.from_pyfile('config.py', silent=True)
    else:
        # 渡されたテスト設定を読み込む
        app.config.from_mapping(test_config)
    
    # ディレクトリの存在確認
    try:
        os.makedirs(app.instance_path, exist_ok=True)
    except OSError:
        pass
    
    # Plamo埋め込みモデルを初期化
    plamo_model_name = app.config['PLAMO_MODEL_NAME']
    app.config['VECTOR_MODEL'] = PlamoEmbedding(plamo_model_name)
    app.logger.info(f"Using Plamo Embedding model: {plamo_model_name}")
    
    # MeCabの初期化
    mecab_dict_path = app.config['MECAB_DICT_PATH']
    app.config['MECAB_TAGGER'] = MeCab.Tagger(f"-d {mecab_dict_path}")
    
    # API認証処理の追加
    @app.before_request
    def authenticate():
        """リクエスト前にAPI認証を行う処理"""
        # API v1エンドポイントのみ認証対象
        if not request.path.startswith('/v1'):
            return None
            
        api_key = request.headers.get('X-API-KEY')
        valid_api_keys = app.config.get('API_KEYS', '').split(',')
        
        if not api_key or api_key not in valid_api_keys:
            abort(401, description='Invalid API Key')
    
    # Blueprintの登録
    app.register_blueprint(parse_bp)
    app.register_blueprint(similarity_bp)
    app.register_blueprint(vector_bp)
    
    # CORSの設定
    from flask_cors import CORS
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    # ルートエンドポイント
    @app.route('/')
    def index():
        """ルートエンドポイント - APIのバージョン情報を返す"""
        return {
            'service': '言語処理API',
            'version': '1.0.0',
        }
    
    return app
