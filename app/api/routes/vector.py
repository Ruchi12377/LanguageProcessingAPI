"""
単語・テキストベクトル計算APIのエンドポイント
"""
from flask import request
from flask_restx import Namespace, Resource, fields
from flask import current_app

from app.api.validators import (
    validate_json_content_type, validate_required_json_fields
)
from app.core.vector_model import get_word_vector, calculate_average_vector

ns = Namespace('vector', description='単語・テキストベクトル計算API')

# 単語ベクトル入力モデル
word_vector_request = ns.model('WordVectorRequest', {
    'word': fields.String(required=True, description='ベクトルを取得する単語')
})

# テキストベクトル入力モデル
texts_vector_request = ns.model('TextsVectorRequest', {
    'texts': fields.List(fields.String, required=True, description='ベクトルを計算するテキストのリスト')
})

# 単語ベクトル成功レスポンスモデル
word_vector_success = ns.model('WordVectorSuccess', {
    'success': fields.Boolean(description='成功フラグ'),
    'vector': fields.List(fields.Float, description='単語ベクトル'),
    'split_words': fields.List(fields.String, description='分割された単語（該当する場合）'),
    'is_compound_word': fields.Boolean(description='複合語フラグ（該当する場合）'),
    'used_words': fields.List(fields.String, description='使用された単語（該当する場合）')
})

# テキストベクトル成功レスポンスモデル
processed_word_info = ns.model('ProcessedWordInfo', {
    'original': fields.String(description='元のテキスト'),
    'used': fields.List(fields.String, description='使用された単語')
})

texts_vector_success = ns.model('TextsVectorSuccess', {
    'success': fields.Boolean(description='成功フラグ'),
    'vector': fields.List(fields.Float, description='テキストベクトル'),
    'warning': fields.String(description='警告メッセージ（該当する場合）'),
    'processed_words': fields.List(fields.Nested(processed_word_info), description='処理された単語情報')
})

# エラーレスポンスモデル
error_response = ns.model('ErrorResponse', {
    'success': fields.Boolean(description='成功フラグ'),
    'error': fields.String(description='エラーメッセージ')
})

@ns.route('/word')
class WordVector(Resource):
    @ns.doc('get_word_vector')
    @ns.expect(word_vector_request)
    @ns.response(200, 'Success', word_vector_success)
    @ns.response(400, 'Bad Request', error_response)
    @validate_json_content_type
    @validate_required_json_fields(['word'])
    def post(self):
        """単語を受け取り、ベクトルを返す。モデルに存在しない場合は分かち書きしてベクトルの平均を返す"""
        data = request.get_json()
        word = data.get("word")

        if not isinstance(word, str):
            return {
                "success": False,
                "error": "word must be a string"
            }

        # モデルとMeCabインスタンスはアプリケーションコンテキストから取得
        model = current_app.config['VECTOR_MODEL']
        mecab_tagger = current_app.config['MECAB_TAGGER']
    
        result = get_word_vector(word, model, mecab_tagger)
        return result

@ns.route('/texts')
class TextsVector(Resource):
    @ns.doc('calculate_texts_vector')
    @ns.expect(texts_vector_request)
    @ns.response(200, 'Success', texts_vector_success)
    @ns.response(400, 'Bad Request', error_response)
    @validate_json_content_type
    @validate_required_json_fields(['texts'])
    def post(self):
        """テキストの配列を受け取り、その平均ベクトルを返す。日本語テキストの場合は分かち書きで処理する"""

        data = request.get_json()
        texts = data.get("texts")

        if not isinstance(texts, list):
            return {
                "success": False,
                "error": "texts must be an array"
            }

        # モデルとMeCabインスタンスはアプリケーションコンテキストから取得
        from flask import current_app
        model = current_app.config['VECTOR_MODEL']
        mecab_tagger = current_app.config['MECAB_TAGGER']
        
        result = calculate_average_vector(texts, model, mecab_tagger)
        return result
