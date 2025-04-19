"""
APIルートの初期化モジュール
"""
from flask import Blueprint

# APIのBlueprint作成
api_bp = Blueprint('api', __name__, url_prefix='/v1')
