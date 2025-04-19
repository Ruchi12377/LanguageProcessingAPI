"""
形態素解析APIのエンドポイント
"""
from flask import request, current_app, Blueprint

from app.utils.text_processing import parse_japanese_text
from app.api.validators import validate_json_content_type, validate_required_json_fields

# ルートの設定
parse_bp = Blueprint('parse', __name__)

@parse_bp.route('/api/v1/parse', methods=['POST'])
@validate_json_content_type
@validate_required_json_fields(['text'])
def parse_text():
    """テキストを形態素解析する"""
    
    data = request.get_json()
    text = data.get('text', '')
    
    # MeCabインスタンスはアプリケーションコンテキストから取得
    mecab_tagger = current_app.config['MECAB_TAGGER']
    
    results = parse_japanese_text(text, mecab_tagger)
    return {'results': results}