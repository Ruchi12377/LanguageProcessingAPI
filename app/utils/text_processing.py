"""
テキスト処理のユーティリティ関数を提供するモジュール
"""
from typing import List, Dict, Any, Optional, Union
import re
import MeCab
from numpy import ndarray, dot, mean
from numpy.linalg import norm

def normalize_vector(vector: ndarray) -> ndarray:
    """ベクトルを正規化する

    Args:
        vector (ndarray): 正規化するベクトル

    Returns:
        ndarray: 正規化されたベクトル
    """
    norm_value = norm(vector)
    return vector / norm_value if norm_value > 0 else vector

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

def parse_japanese_text(text: str, mecab_tagger: Optional[MeCab.Tagger] = None) -> List[Dict[str, str]]:
    """日本語のテキストを形態素解析する

    Args:
        text (str): 解析する日本語テキスト
        mecab_tagger (Optional[MeCab.Tagger]): MeCabのTaggerオブジェクト。Noneの場合は新しく作成します。

    Returns:
        List[Dict[str, str]]: 形態素解析の結果（surface: 表層形, feature: 品詞情報など）
    """
    if mecab_tagger is None:
        mecab_tagger = MeCab.Tagger()
        
    parsed = mecab_tagger.parse(text)
    return [{"surface": line.split("\t")[0], "feature": line.split("\t")[1]}
            for line in parsed.split("\n") if line and "\t" in line]

def split_by_caps_underscores_spaces(word: str) -> List[str]:
    """英語の単語を大文字、アンダースコア、空白で分割する。
    
    Args:
        word (str): 分割する単語
        
    Returns:
        List[str]: 分割された単語のリスト（小文字化された状態）
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

def calculate_similarity_between_vectors(vector1: ndarray, vector2: ndarray) -> float:
    """2つのベクトル間の類似度を計算する
    
    Args:
        vector1 (ndarray): 1つ目のベクトル
        vector2 (ndarray): 2つ目のベクトル
        
    Returns:
        float: コサイン類似度
    """
    return float(dot(vector1, vector2) / (norm(vector1) * norm(vector2)))

def extract_words_from_text(text: str, mecab_tagger: MeCab.Tagger) -> Dict[str, Any]:
    """テキストから名詞、形容詞、動詞を抽出する
    
    Args:
        text (str): 抽出する対象のテキスト
        mecab_tagger (MeCab.Tagger): MeCabのTaggerオブジェクト
        
    Returns:
        Dict[str, Any]: 抽出された名詞、形容詞、動詞（基本形）と、順序を保持した単語リスト
    """
    parsed = mecab_tagger.parse(text)
    words = {
        "nouns": [],
        "adjectives": [],
        "verbs": [],
        "ordered_words": []  # 順序を保持するリスト
    }
    
    for line in parsed.split('\n'):
        if line == 'EOS' or not line:
            continue
        
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        
        surface = parts[0]
        features = parts[1].split(',')
        
        if len(features) < 7:  # 基本形の情報が含まれていない場合はスキップ
            continue
        
        word_info = None
            
        # 品詞に基づいて単語を分類
        if features[0] == '名詞':
            # 特殊な名詞（代名詞、特殊）をスキップ
            if len(features) > 1 and features[2] in ['代名詞', '特殊']:
                continue
            words["nouns"].append(surface)
            word_info = {"word": surface, "type": "noun"}
            
        elif features[0] == '形容詞':
            # 基本形が存在する場合はそれを使用、なければ表層形を使用
            base_form = features[6] if features[6] != '*' else surface
            words["adjectives"].append(base_form)
            word_info = {"word": base_form, "type": "adjective"}
            
        elif features[0] == '動詞':
            # 基本形が存在する場合はそれを使用、なければ表層形を使用
            base_form = features[6] if features[6] != '*' else surface
            words["verbs"].append(base_form)
            word_info = {"word": base_form, "type": "verb"}
            
        # 有効な単語があれば順序付きリストに追加
        if word_info:
            words["ordered_words"].append(word_info)
    
    return words
