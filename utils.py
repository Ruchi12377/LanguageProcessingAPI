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

def get_word_vector(word: str, model, mecab_tagger: MeCab.Tagger) -> Dict[str, Any]:
    """単語のベクトルを取得する。単語がモデルに存在しない場合は分かち書きして名詞や形容詞を取り出し平均ベクトルを返す
    
    Args:
        word (str): 単語
        model: 単語ベクトルモデル
        mecab_tagger (MeCab.Tagger): MeCabのTaggerオブジェクト
        
    Returns:
        Dict[str, Any]: 結果の辞書
    """
    # 単語がモデルに存在するかチェック
    if word in model:
        vector = model[word]
        return {
            "success": True,
            "vector": vector.tolist(),
        }
    
    # 英語の単語の場合、分割を試みる
    if any(c.isupper() for c in word) or '_' in word or ' ' in word:
        split_words = split_by_caps_underscores_spaces(word)
        if all(w in model for w in split_words) and split_words:
            vector = mean([model[w] for w in split_words], axis=0)
            normalized_vector = normalize_vector(vector)
            
            return {
                "success": True,
                "vector": normalized_vector.tolist(),
                "split_words": split_words,
                "is_compound_word": True
            }
    
    # 単語をMeCabで分かち書きし、名詞、形容詞、動詞を抽出
    extracted = extract_words_from_text(word, mecab_tagger)
    ordered_words = extracted["ordered_words"]
    
    # 処理対象の単語リストを作成（元の順序を保持）
    words_to_process = [word_info["word"] for word_info in ordered_words]
    
    # 抽出された単語がない場合
    if not words_to_process:
        return {
            "success": False,
            "error": f"No nouns, adjectives, or verbs found in word: {word}"
        }
    
    # 抽出された単語のうち、モデルに存在するものだけを使用
    valid_words = [w for w in words_to_process if w in model]
    
    # 有効な単語が見つからなかった場合
    if not valid_words:
        return {
            "success": False,
            "error": f"No words found in model vocabulary from word: {word}",
        }
    
    # 単語のベクトルの平均を計算し、正規化
    vectors = [model[w] for w in valid_words]
    average_vector = mean(vectors, axis=0)
    normalized_vector = normalize_vector(average_vector)
    
    return {
        "success": True,
        "vector": normalized_vector.tolist(),
        "used_words": valid_words
    }

def calculate_average_vector(texts: List[str], model, mecab_tagger: Optional[MeCab.Tagger] = None) -> Dict[str, Any]:
    """テキストのリストから平均ベクトルを計算する。日本語テキストの場合は分かち書きで処理する
    
    Args:
        texts (List[str]): テキストのリスト
        model: 単語ベクトルモデル
        mecab_tagger (Optional[MeCab.Tagger]): MeCabのTaggerオブジェクト
        
    Returns:
        Dict[str, Any]: 結果の辞書（成功の場合はベクトルと成功フラグ、失敗の場合はエラーメッセージ）
    """
    if not texts:
        return {"success": False, "error": "Empty text list"}
    
    # 各テキストのベクトルを計算
    vectors = []
    missing_words = []
    processed_words = []
    
    for text in texts:
        # まず、直接単語ベクトルを取得できるか試みる
        vector = calculate_vector_for_word(text, model)
        if vector is not None:
            vectors.append(vector)
            processed_words.append({"original": text, "used": [text]})
            continue
        
        # 直接取得できない場合、日本語テキストとして処理を試みる
        if mecab_tagger:
            # テキストから名詞、形容詞、動詞を抽出
            extracted = extract_words_from_text(text, mecab_tagger)
            words_to_process = [word_info["word"] for word_info in extracted["ordered_words"]]
            
            # 抽出された単語がある場合
            if words_to_process:
                # モデルに存在する単語だけを使用
                valid_words = [w for w in words_to_process if w in model]
                
                if valid_words:
                    # 単語のベクトルの平均を計算
                    text_vectors = [model[w] for w in valid_words]
                    text_avg_vector = mean(text_vectors, axis=0)
                    vectors.append(text_avg_vector)
                    processed_words.append({"original": text, "used": valid_words})
                    continue
        
        # 処理できなかった単語は記録
        missing_words.append(text)
    
    # ベクトルが見つからなかった場合
    if not vectors:
        return {
            "success": False,
            "error": f"No words found in model vocabulary: {', '.join(missing_words)}"
        }
    
    # 平均ベクトルを計算し、正規化
    average_vector = mean(vectors, axis=0)
    normalized_vector = normalize_vector(average_vector)
    
    result = {
        "success": True,
        "vector": normalized_vector.tolist(),  # JSONシリアライズ可能な形式に変換
    }
    
    # 一部の単語が見つからなかった場合は警告を追加
    if missing_words:
        result["warning"] = f"Some words not found in model vocabulary: {', '.join(missing_words)}"
    
    # 処理された単語情報を追加
    if processed_words:
        result["processed_words"] = processed_words
        
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
