"""
アプリケーション初期化モジュール
"""
import os
from flask import Flask, request, abort
import MeCab

from app.api import api_bp
from app.core.vector_model import VectorModel

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
        MODEL_PATH=os.getenv('MODEL_PATH', './model_gensim_norm'),
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
    
    # モデルの初期化
    model_path = app.config['MODEL_PATH']
    app.config['VECTOR_MODEL'] = VectorModel(model_path)
    
    # MeCabの初期化
    mecab_dict_path = app.config['MECAB_DICT_PATH']
    app.config['MECAB_TAGGER'] = MeCab.Tagger(f"-d {mecab_dict_path}")
    
    # API認証処理の追加
    @app.before_request
    def authenticate():
        """リクエスト前にAPI認証を行う処理（swagger.jsonエンドポイントとドキュメントを除く）"""
        # swagger.json エンドポイントは認証から除外
        if request.path.endswith('/swagger.json'):
            return None
            
        # docs パスも認証から除外
        if '/docs' in request.path:
            return None
            
        # API v1エンドポイントのみ認証対象
        if not request.path.startswith('/api/v1'):
            return None
            
        api_key = request.headers.get('X-API-KEY')
        valid_api_keys = app.config.get('API_KEYS', '').split(',')
        
        if not api_key or api_key not in valid_api_keys:
            abort(401, description='Invalid API Key')
    
    # Blueprintの登録
    app.register_blueprint(api_bp)
    
    # CORSの設定
    from flask_cors import CORS
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    # ルートエンドポイント
    @app.route('/')
    def index():
        """ルートエンドポイント - APIのバージョンとドキュメントへのリンクを返す"""
        return {
            'service': '言語処理API',
            'version': '1.0.0',
        }
    
    return app
