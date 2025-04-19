"""
単語・テキストベクトル計算APIのエンドポイント
"""
from flask import request, current_app, Blueprint

from app.api.validators import (
    validate_json_content_type, validate_required_json_fields
)
from app.core.vector_model import get_word_vector, calculate_average_vector

# ルートの設定
vector_bp = Blueprint('vector', __name__)

@vector_bp.route('/v1/vector/word', methods=['POST'])
@validate_json_content_type
@validate_required_json_fields(['word'])
def get_word_vector_route():
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

@vector_bp.route('/v1/vector/texts', methods=['POST'])
@validate_json_content_type
@validate_required_json_fields(['texts'])
def calculate_texts_vector():
    """テキストの配列を受け取り、その平均ベクトルを返す。日本語テキストの場合は分かち書きで処理する"""
    data = request.get_json()
    texts = data.get("texts")

    if not isinstance(texts, list):
        return {
            "success": False,
            "error": "texts must be an array"
        }

    # モデルとMeCabインスタンスはアプリケーションコンテキストから取得
    model = current_app.config['VECTOR_MODEL']
    mecab_tagger = current_app.config['MECAB_TAGGER']
    
    result = calculate_average_vector(texts, model, mecab_tagger)
    return result
