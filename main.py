from dotenv import load_dotenv
from flask import Flask, request, jsonify
import os
import gensim
import MeCab
from typing import List, Dict, Any, Tuple, Optional
from flask_cors import CORS

# 環境変数の読み込み
load_dotenv()

# アプリケーション初期化
app = Flask(__name__)
mecab = MeCab.Tagger()

# モデル読み込み
MODEL_PATH = "./model_gensim_norm"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model = gensim.models.KeyedVectors.load(MODEL_PATH, mmap='r')

# CORSの設定
allowed_domains = os.getenv("ALLOWED_DOMAIN", "example.com").split(",")
CORS(app, resources={r"/*": {"origins": allowed_domains}})

def validate_request() -> Optional[Tuple[Dict[str, str], int]]:
    """リクエストの検証を行う共通関数

    Returns:
        Optional[Tuple[Dict[str, str], int]]: エラーがある場合はエラーレスポンスとステータスコード、なければNone
    """
    referrer = request.referrer
    if not referrer or not any(domain in referrer for domain in allowed_domains):
        return {"error": "Access denied"}, 403
    return None

@app.route("/parse", methods=["GET"])
def parse() -> Dict[str, Any]:
    """テキストを形態素解析する

    Returns:
        Dict[str, Any]: 形態素解析の結果
    """
    error_response = validate_request()
    if error_response:
        return jsonify(error_response[0]), error_response[1]

    text = request.args.get("text", "")
    parsed = mecab.parse(text)
    result = [{"surface": line.split("\t")[0], "feature": line.split("\t")[1]}
              for line in parsed.split("\n") if line and "\t" in line]
    return jsonify(result)

@app.route("/distance", methods=["GET"])
def distance() -> Dict[str, Any]:
    """単語間の類似度を計算する

    Returns:
        Dict[str, Any]: 類似度計算の結果
    """
    error_response = validate_request()
    if error_response:
        return jsonify(error_response[0]), error_response[1]

    if request.headers.get("Content-Type") != "application/json":
        return jsonify({
            "pairs": [],
            "error": "Content-Type is wrong"
        })

    pairs = request.get_json().get("pairs")
    if pairs is None:
        return jsonify({
            "pairs": [],
            "error": "Missing pairs"
        })

    result = process_word_pairs(pairs)

    return jsonify({
        "pairs": result,
        "error": ""
    })

def process_word_pairs(pairs: List) -> List[Dict[str, str]]:
    """単語ペアの処理を行い類似度を計算する

    Args:
        pairs (List): 単語ペアのリスト

    Returns:
        List[Dict[str, str]]: 類似度計算結果のリスト
    """
    result = []
    for pair in pairs:
        word1 = pair[0] if pair[0] is not None else ""
        word2 = pair[1] if pair[1] is not None else ""
        if word1 == "" or word2 == "":
            continue

        similarity = ""
        errorMessage = ""
        isWord1InVocab = word1 in model
        isWord2InVocab = word2 in model
        if isWord1InVocab and isWord2InVocab:
            similarity = str(model.similarity(word1, word2))

        if not isWord1InVocab and not isWord2InVocab:
            errorMessage = f"'{word1}' and '{word2}' not found in the vocabulary"
        elif not isWord1InVocab:
            errorMessage = f"'{word1}' not found in the vocabulary"
        elif not isWord2InVocab:
            errorMessage = f"'{word2}' not found in the vocabulary"

        result.append({
            "word1": word1,
            "word2": word2,
            "similarity": similarity,
            "error": errorMessage
        })

    return result

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
