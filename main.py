# TODO: 日本語の単語を分割する関数を追加する

from dotenv import load_dotenv
from flask import Flask, request, jsonify
import os
import gensim
import MeCab
from typing import List, Dict, Any, Tuple, Optional
from flask_cors import CORS
import re

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

@app.route("/distance", methods=["POST"])
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

def split_by_capitals(word: str) -> List[str]:
    """英語の単語を大文字で分割する。ただし、大文字が連続している場合（acronym）は分割しない。
    
    Args:
        word (str): 分割する単語
        
    Returns:
        List[str]: 分割された単語のリスト
    """
    if not word:
        return []
        
    # 連続する大文字を一つのグループとし、その後の小文字も含めたパターン
    pattern = r'[A-Z]+[a-z]*'
    
    # 単語の先頭が小文字の場合、それを最初の部分として取得
    first_part = ""
    if word and word[0].islower():
        match = re.match(r'^[a-z]+', word)
        if match:
            first_part = match.group(0)
            word = word[len(first_part):]
    
    # 残りの部分を大文字のパターンで分割
    parts = re.findall(pattern, word)
    
    # 先頭の小文字部分があれば追加
    result = []
    if first_part:
        result.append(first_part)
    result.extend(parts)
    
    return [p for p in result if p]

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

        similarity = 0
        errorMessage = ""
        isWord1InVocab = word1 in model
        isWord2InVocab = word2 in model
        
        # 元の単語がボキャブラリーにあればそのまま類似度を計算
        if isWord1InVocab and isWord2InVocab:
            try:
                similarity = float(model.similarity(word1, word2))
            except:
                errorMessage = f"Failed to calculate similarity between '{word1}' and '{word2}'"
        else:
            # ボキャブラリーにない単語の処理
            processed_word1 = word1
            processed_word2 = word2
            
            # 英語の単語で辞書にない場合は大文字で分割を試みる
            if not isWord1InVocab and any(c.isupper() for c in word1):
                split_words1 = split_by_capitals(word1)
                # 分割された単語がボキャブラリーにあるかチェック
                if all(w in model for w in split_words1):
                    processed_word1 = split_words1
                    isWord1InVocab = True
                
            if not isWord2InVocab and any(c.isupper() for c in word2):
                split_words2 = split_by_capitals(word2)
                # 分割された単語がボキャブラリーにあるかチェック
                if all(w in model for w in split_words2):
                    processed_word2 = split_words2
                    isWord2InVocab = True
                
            # 分割後の単語で類似度計算
            if isWord1InVocab and isWord2InVocab:
                try:
                    # 分割された場合、分割された単語の平均ベクトルを使用
                    if isinstance(processed_word1, list):
                        vector1 = sum(model[w] for w in processed_word1) / len(processed_word1)
                    else:
                        vector1 = model[processed_word1]
                    
                    if isinstance(processed_word2, list):
                        vector2 = sum(model[w] for w in processed_word2) / len(processed_word2)
                    else:
                        vector2 = model[processed_word2]
                    
                    # cosine_similarityでベクトル間の類似度を計算
                    from numpy import dot
                    from numpy.linalg import norm
                    similarity = float(dot(vector1, vector2) / (norm(vector1) * norm(vector2)))
                except Exception as e:
                    errorMessage = f"Failed to calculate similarity: {str(e)}"
        
        # エラーメッセージの設定
        if not errorMessage:
            if not isWord1InVocab and not isWord2InVocab:
                errorMessage = f"'{word1}' and '{word2}' not found in the vocabulary"
            elif not isWord1InVocab:
                errorMessage = f"'{word1}' not found in the vocabulary"
            elif not isWord2InVocab:
                errorMessage = f"'{word2}' not found in the vocabulary"

        result.append({
            "word1": word1,
            "word2": word2,
            "similarity": str(similarity) if isinstance(similarity, float) else similarity,
            "error": errorMessage
        })

    return result

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
