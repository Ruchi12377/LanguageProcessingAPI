"""
単語類似度計算APIのエンドポイント
"""
from flask import request
from flask_restx import Namespace, Resource, fields

from app.core.vector_model import process_word_pairs

ns = Namespace('similarity', description='単語類似度計算API')

# 入力モデルの定義
word_pair_model = ns.model('WordPair', {
    'word1': fields.String(required=True, description='1つ目の単語'),
    'word2': fields.String(required=True, description='2つ目の単語')
})

similarity_request = ns.model('SimilarityRequest', {
    'pairs': fields.List(fields.List(fields.String), required=True, description='単語ペアのリスト')
})

# レスポンスモデルの定義
similarity_result = ns.model('SimilarityResult', {
    'word1': fields.String(description='1つ目の単語'),
    'word2': fields.String(description='2つ目の単語'),
    'similarity': fields.Float(description='類似度（0～1）'),
    'error': fields.String(description='エラーメッセージ（エラーがある場合）')
})

similarity_response = ns.model('SimilarityResponse', {
    'pairs': fields.List(fields.Nested(similarity_result), description='類似度計算結果'),
    'error': fields.String(description='全体エラーメッセージ（エラーがある場合）')
})

@ns.route('')
class Similarity(Resource):
    @ns.doc('calculate_similarity')
    @ns.expect(similarity_request)
    @ns.marshal_with(similarity_response)
    def post(self):
        """単語間の類似度を計算する"""
        data = request.get_json()
        pairs = data.get("pairs", [])
        if not pairs:
            return {
                "pairs": [],
                "error": "Missing pairs"
            }

        # モデルはアプリケーションコンテキストから取得（正しい方法）
        from flask import current_app
        model = current_app.config['VECTOR_MODEL']
        result = process_word_pairs(pairs, model)

        return {
            "pairs": result,
            "error": ""
        }
