"""
形態素解析APIのエンドポイント
"""
from flask import request, current_app
from flask_restx import Namespace, Resource, fields

from app.core.vector_model import VectorModel
from app.utils.text_processing import parse_japanese_text

ns = Namespace('parse', description='テキスト形態素解析API')

# 入力モデルの定義
parse_model = ns.model('ParseRequest', {
    'text': fields.String(required=True, description='解析する日本語テキスト')
})

# レスポンスモデルの定義
parse_item = ns.model('ParseItem', {
    'surface': fields.String(description='表層形'),
    'feature': fields.String(description='品詞情報')
})

parse_response = ns.model('ParseResponse', {
    'results': fields.List(fields.Nested(parse_item), description='形態素解析の結果')
})

@ns.route('')
class Parse(Resource):
    @ns.doc('parse_text')
    @ns.expect(parse_model, validate=True)
    @ns.marshal_with(parse_response)
    def post(self):
        """テキストを形態素解析する"""
        
        data = request.get_json()
        text = data.get('text', '')
        
        # MeCabインスタンスはアプリケーションコンテキストから取得
        mecab_tagger = current_app.config['MECAB_TAGGER']
        
        results = parse_japanese_text(text, mecab_tagger)
        return {'results': results}