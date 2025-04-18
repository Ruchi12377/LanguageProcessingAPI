from typing import List, Dict, Any, Optional, Union
import re
from numpy import ndarray, dot, mean
from numpy.linalg import norm

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

def split_by_caps_underscores_spaces(word: str) -> List[str]:
    """英語の単語を大文字、アンダースコア、空白で分割する。
    
    Args:
        word (str): 分割する単語
        
    Returns:
        List[str]: 分割された単語のリスト
    """
    if not word:
        return []
    
    # まず、アンダースコアと空白で分割
    parts = re.split(r'[_\s]+', word)
    result = []
    
    # 各部分をさらに大文字で分割
    for part in parts:
        if not part:
            continue
            
        # 先頭が小文字の場合の処理
        first_part = ""
        current_part = part
        if part and part[0].islower():
            match = re.match(r'^[a-z0-9]+', part)
            if match:
                first_part = match.group(0)
                current_part = part[len(first_part):]
        
        # 大文字で始まる部分を分割
        cap_parts = re.findall(r'[A-Z]+[a-z0-9]*', current_part)
        
        # 結果に追加
        if first_part:
            result.append(first_part)
        result.extend(cap_parts)
    
    return [p.lower() for p in result if p]

def calculate_vector_for_word(word: str, model, try_split: bool = True) -> Optional[ndarray]:
    """単語のベクトルを計算する。モデルにない場合は分割を試みる。
    
    Args:
        word (str): 単語
        model: 単語ベクトルモデル
        try_split (bool): 分割を試みるかどうか
        
    Returns:
        Optional[ndarray]: 単語のベクトル。単語がモデルに見つからない場合はNone
    """
    if word in model:
        return model[word]
    
    if try_split and (any(c.isupper() for c in word) or '_' in word or ' ' in word):
        split_words = split_by_caps_underscores_spaces(word)
        if all(w in model for w in split_words) and split_words:
            return mean([model[w] for w in split_words], axis=0)
    
    return None

def calculate_average_vector(texts: List[str], model) -> Dict[str, Any]:
    """テキストのリストから平均ベクトルを計算する
    
    Args:
        texts (List[str]): テキストのリスト
        model: 単語ベクトルモデル
        
    Returns:
        Dict[str, Any]: 結果の辞書（成功の場合はベクトルと成功フラグ、失敗の場合はエラーメッセージ）
    """
    if not texts:
        return {"success": False, "error": "Empty text list"}
    
    # 各テキストのベクトルを計算
    vectors = []
    missing_words = []
    
    for text in texts:
        vector = calculate_vector_for_word(text, model)
        if vector is not None:
            vectors.append(vector)
        else:
            missing_words.append(text)
    
    # ベクトルが見つからなかった場合
    if not vectors:
        return {
            "success": False,
            "error": f"No words found in model vocabulary: {', '.join(missing_words)}"
        }
    
    # 平均ベクトルを計算
    average_vector = mean(vectors, axis=0)
    
    # 正規化
    norm_value = norm(average_vector)
    if norm_value > 0:
        normalized_vector = average_vector / norm_value
    else:
        normalized_vector = average_vector
    
    result = {
        "success": True,
        "vector": normalized_vector.tolist(),  # JSONシリアライズ可能な形式に変換
    }
    
    # 一部の単語が見つからなかった場合は警告を追加
    if missing_words:
        result["warning"] = f"Some words not found in model vocabulary: {', '.join(missing_words)}"
        
    return result

def calculate_similarity_between_vectors(vector1: ndarray, vector2: ndarray) -> float:
    """2つのベクトル間の類似度を計算する
    
    Args:
        vector1 (ndarray): 1つ目のベクトル
        vector2 (ndarray): 2つ目のベクトル
        
    Returns:
        float: コサイン類似度
    """
    return float(dot(vector1, vector2) / (norm(vector1) * norm(vector2)))
