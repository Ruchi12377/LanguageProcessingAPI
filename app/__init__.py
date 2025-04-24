"""
アプリケーション初期化モジュール
"""
import os
from flask import Flask, request, abort

# Blueprintのインポート
from app.api.routes.similarity import similarity_bp
from app.api.routes.vector import vector_bp
from app.core.plamo_embedding import PlamoEmbedding
from app.utils.vector_cache import VectorCache

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
        API_KEYS=os.getenv('API_KEYS', 'default-key'),
        VECTOR_CACHE_PATH=os.getenv('VECTOR_CACHE_PATH', None),
        VECTOR_CACHE_ENABLED=os.getenv('VECTOR_CACHE_ENABLED', '1') == '1'
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
    app.config['VECTOR_MODEL'] = PlamoEmbedding(plamo_model_name, use_fp16=True)
    app.logger.info(f"Using Plamo Embedding model: {plamo_model_name}")
    
    # ベクトルキャッシュの初期化（設定で有効の場合）
    if app.config['VECTOR_CACHE_ENABLED']:
        app.logger.info("Initializing vector cache")
        app.config['VECTOR_CACHE'] = VectorCache(app.config['VECTOR_CACHE_PATH'])
        
        # キャッシュ統計情報をログに出力
        try:
            stats = app.config['VECTOR_CACHE'].get_stats()
            app.logger.info(f"Vector cache contains {stats['total_entries']} entries")
        except Exception as e:
            app.logger.error(f"Error getting cache stats: {str(e)}")
    
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
