"""
単語類似度計算APIのエンドポイント
"""
from flask import request, current_app, Blueprint

from app.core.vector_model import process_word_pairs
from app.api.validators import validate_json_content_type, validate_required_json_fields

# ルートの設定
similarity_bp = Blueprint('similarity', __name__)

@similarity_bp.route('/v1/similarity', methods=['POST'])
@validate_json_content_type
@validate_required_json_fields(['pairs'])
def calculate_similarity():
    """単語間の類似度を計算する"""
    data = request.get_json()
    pairs = data.get("pairs", [])
    if not pairs:
        return {
            "pairs": [],
            "error": "Missing pairs"
        }

    # モデルはアプリケーションコンテキストから取得
    model = current_app.config['VECTOR_MODEL']
    result = process_word_pairs(pairs, model)

    return {
        "pairs": result,
        "error": ""
    }
