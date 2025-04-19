"""
APIルートの初期化モジュール
"""
from flask import Blueprint
from flask_restx import Api

# APIのBlueprint作成
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# Flask-RESTxのAPIインスタンス作成
authorizations = {
    'apikey': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'X-API-KEY'
    }
}

api = Api(
    api_bp,
    version='1.0',
    title='言語処理API',
    description='テキスト解析と単語ベクトル処理のためのAPI',
    doc='/docs',
    authorizations=authorizations
)

# APIのNamespaceをインポート
from app.api.routes.parse import ns as parse_ns
from app.api.routes.similarity import ns as similarity_ns
from app.api.routes.vector import ns as vector_ns

# APIにNamespaceを追加
api.add_namespace(parse_ns)
api.add_namespace(similarity_ns)
api.add_namespace(vector_ns)
