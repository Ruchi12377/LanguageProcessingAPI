"""
APIリクエストのバリデーションと共通処理を提供するモジュール
"""
from flask import request
from functools import wraps

def validate_json_content_type(f):
    """JSONコンテンツタイプのバリデーションを行うデコレーター

    Args:
        f: デコレートする関数

    Returns:
        decorated_function: デコレートされた関数
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.headers.get("Content-Type") != "application/json":
            return {"success": False, "error": "Content-Type must be application/json"}, 400
        return f(*args, **kwargs)
    return decorated_function

def validate_required_json_fields(required_fields):
    """必須JSONフィールドのバリデーションを行うデコレーター

    Args:
        required_fields: 必須フィールドのリスト

    Returns:
        decorator: デコレーター関数
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            data = request.get_json()
            if not data:
                return {"success": False, "error": "No JSON data provided"}, 400
            
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return {
                    "success": False,
                    "error": f"Missing required fields: {', '.join(missing_fields)}"
                }, 400
                
            return f(*args, **kwargs)
        return decorated_function
    return decorator
