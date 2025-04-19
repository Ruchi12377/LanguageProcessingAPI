"""
テキスト処理のユーティリティ関数を提供するモジュール
"""
from typing import List, Dict, Any, Optional, Union
import re
import MeCab
import numpy as np
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
        model: 単語ベクトルモデル（PlamoEmbeddingまたはVectorModel）
        try_split (bool): 分割を試みるかどうか
        
    Returns:
        Optional[ndarray]: 単語のベクトル。単語がモデルに見つからない場合はNone
    """
    # PlamoEmbeddingの場合は直接ベクトル取得（すべての単語に対応可能）
    from app.core.plamo_embedding import PlamoEmbedding
    if isinstance(model, PlamoEmbedding):
        if model.is_initialized:
            return model.get_vector(word)
        return None
        
    # Gensimモデルの場合
    if word in model:
        return model.get_vector(word)
    
    if try_split and (any(c.isupper() for c in word) or '_' in word or ' ' in word):
        split_words = split_by_caps_underscores_spaces(word)
        if all(w in model for w in split_words) and split_words:
            return mean([model.get_vector(w) for w in split_words], axis=0)
    
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

def get_word_vector(word: str, model, mecab_tagger: Optional[MeCab.Tagger] = None) -> Dict[str, Any]:
    """単語のベクトルを取得する。モデルに応じて適切な処理を行う
    
    Args:
        word (str): 単語
        model: ベクトルモデル（PlamoEmbeddingまたはVectorModel）
        mecab_tagger (Optional[MeCab.Tagger]): MeCabのTaggerオブジェクト
        
    Returns:
        Dict[str, Any]: 結果の辞書
    """
    # PlamoEmbeddingモデルの場合
    from app.core.plamo_embedding import PlamoEmbedding
    if isinstance(model, PlamoEmbedding):
        if not model.is_initialized:
            return {"success": False, "error": "Plamo embedding model is not initialized"}
        try:
            # Plamoはどんな単語も直接ベクトル化できる
            vector = model.get_vector(word)
            return {
                "success": True,
                "vector": vector.tolist(),
                "model": "plamo-embedding"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error computing embedding with Plamo model: {str(e)}"
            }

    # 以下はGensimモデルの処理
    # 単語がモデルに存在するかチェック
    if word in model:
        vector = model.get_vector(word)
        return {
            "success": True,
            "vector": vector.tolist(),
        }
    
    # 英語の単語の場合、分割を試みる
    if any(c.isupper() for c in word) or '_' in word or ' ' in word:
        split_words = split_by_caps_underscores_spaces(word)
        if all(w in model for w in split_words) and split_words:
            vector = mean([model.get_vector(w) for w in split_words], axis=0)
            normalized_vector = normalize_vector(vector)
            
            return {
                "success": True,
                "vector": normalized_vector.tolist(),
                "split_words": split_words,
                "is_compound_word": True
            }
    
    # 単語をMeCabで分かち書きし、名詞、形容詞、動詞を抽出
    if mecab_tagger:
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
        vectors = [model.get_vector(w) for w in valid_words]
        average_vector = mean(vectors, axis=0)
        normalized_vector = normalize_vector(average_vector)
        
        return {
            "success": True,
            "vector": normalized_vector.tolist(),
            "used_words": valid_words
        }
    
    return {
        "success": False,
        "error": f"Could not process word: {word}"
    }

def calculate_average_vector(texts: List[str], model, mecab_tagger: Optional[MeCab.Tagger] = None) -> Dict[str, Any]:
    """テキストのリストから平均ベクトルを計算する。
    PlamoEmbeddingモデルが利用可能な場合はそれを使用し、
    そうでない場合は従来の分かち書きと単語ベクトルの平均を計算する。
    
    Args:
        texts (List[str]): テキストのリスト
        model: ベクトルモデル（VectorModelまたはPlamoEmbedding）
        mecab_tagger (Optional[MeCab.Tagger]): MeCabのTaggerオブジェクト
        
    Returns:
        Dict[str, Any]: 結果の辞書（成功の場合はベクトルと成功フラグ、失敗の場合はエラーメッセージ）
    """
    if not texts:
        return {"success": False, "error": "Empty text list"}
    
    # PlamoEmbeddingモデルの場合
    from app.core.plamo_embedding import PlamoEmbedding
    if isinstance(model, PlamoEmbedding):
        if not model.is_initialized:
            return {"success": False, "error": "Plamo embedding model is not initialized"}
        
        try:
            # テキストを埋め込みベクトルに変換
            embeddings = model.encode_documents(texts)
            
            # 平均ベクトルを計算（複数テキストの場合）
            if len(embeddings) > 1:
                average_vector = mean(embeddings, axis=0)
            else:
                average_vector = embeddings[0]
            
            # 正規化
            normalized_vector = normalize_vector(average_vector)
            
            return {
                "success": True,
                "vector": normalized_vector.tolist(),  # JSONシリアライズ可能な形式に変換
                "model": "plamo-embedding"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error computing embeddings with Plamo model: {str(e)}"
            }
    
    # 従来のVectorModelの場合
    vectors = []
    missing_words = []
    processed_words = []
    
    for text in texts:
        # 空白とアンダースコアで分割
        is_compound = False
        compound_parts = []
        
        # 空白で分割
        space_parts = text.split()
        if len(space_parts) > 1:
            is_compound = True
        
        # 各部分をさらにアンダースコアで分割
        all_parts = []
        for part in space_parts:
            underscore_parts = part.split('_')
            if len(underscore_parts) > 1:
                is_compound = True
            all_parts.extend([p.strip() for p in underscore_parts if p.strip()])
            
        # 分割がなかった場合はオリジナルのテキストを使用
        if not all_parts:
            all_parts = [text]
            
        # 分割された部分を保存
        compound_parts = all_parts
            
        # 分割したパーツのベクトルを計算
        part_vectors = []
        valid_parts = []
        
        for part in compound_parts:
            # まず、直接単語ベクトルを取得できるか試みる
            vector = calculate_vector_for_word(part, model)
            if vector is not None:
                part_vectors.append(vector)
                valid_parts.append(part)
                continue
            
            # 直接取得できない場合、日本語テキストとして処理を試みる
            if mecab_tagger:
                extracted = extract_words_from_text(part, mecab_tagger)
                words_to_process = [word_info["word"] for word_info in extracted["ordered_words"]]
                
                # 抽出された単語がある場合
                if words_to_process:
                    valid_words = [w for w in words_to_process if w in model]
                    
                    if valid_words:
                        # 単語のベクトルの平均を計算
                        word_vectors = [model.get_vector(w) for w in valid_words]
                        word_avg_vector = mean(word_vectors, axis=0)
                        part_vectors.append(word_avg_vector)
                        valid_parts.extend(valid_words)
        
        # パーツのベクトルが見つかった場合
        if part_vectors:
            # 複合語の場合は平均ベクトルを計算
            if is_compound or len(part_vectors) > 1:
                text_avg_vector = mean(part_vectors, axis=0)
                vectors.append(text_avg_vector)
                processed_words.append({
                    "original": text,
                    "used": valid_parts,
                    "is_compound": True
                })
            else:
                # 単一の単語の場合はそのベクトルを使用
                vectors.append(part_vectors[0])
                processed_words.append({
                    "original": text,
                    "used": valid_parts
                })
        else:
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
        "model": "gensim"
    }
    
    # 一部の単語が見つからなかった場合は警告を追加
    if missing_words:
        result["warning"] = f"Some words not found in model vocabulary: {', '.join(missing_words)}"
    
    # 処理された単語情報を追加
    if processed_words:
        result["processed_words"] = processed_words
        
    return result

def process_word_pairs(pairs: List, model) -> List[Dict[str, Any]]:
    """単語ペアの処理を行い類似度を計算する
    
    Args:
        pairs (List): 単語ペアのリスト
        model: ベクトルモデル（PlamoEmbeddingまたはVectorModel）
        
    Returns:
        List[Dict[str, str]]: 類似度計算結果のリスト
    """
    result = []
    from app.core.plamo_embedding import PlamoEmbedding
    using_plamo = isinstance(model, PlamoEmbedding)
    
    for pair in pairs:
        word1 = pair[0] if pair[0] is not None else ""
        word2 = pair[1] if pair[1] is not None else ""
        if word1 == "" or word2 == "":
            continue

        similarity = 0
        errorMessage = ""

        # PlamoEmbeddingモデルの場合
        if using_plamo:
            if not model.is_initialized:
                errorMessage = "Plamo embedding model is not initialized"
            else:
                try:
                    # PlamoEmbeddingの場合は直接テキスト間の類似度を計算できる
                    similarity = model.similarity(word1, word2)
                except Exception as e:
                    errorMessage = f"Failed to calculate similarity with Plamo model: {str(e)}"
        # Gensimモデルの場合
        else:
            isWord1InVocab = word1 in model
            isWord2InVocab = word2 in model
            
            # 元の単語がボキャブラリーにあればそのまま類似度を計算
            if isWord1InVocab and isWord2InVocab:
                try:
                    similarity = model.similarity(word1, word2)
                except Exception:
                    errorMessage = f"Failed to calculate similarity between '{word1}' and '{word2}'"
            else:
                # ボキャブラリーにない単語の処理
                processed_word1 = word1
                processed_word2 = word2
                
                # 英語の単語で辞書にない場合は大文字、アンダースコア、空白で分割を試みる
                if not isWord1InVocab and (any(c.isupper() for c in word1) or '_' in word1 or ' ' in word1):
                    split_words1 = split_by_caps_underscores_spaces(word1)
                    # 分割された単語がボキャブラリーにあるかチェック
                    if all(w in model for w in split_words1):
                        processed_word1 = split_words1
                        isWord1InVocab = True
                    
                if not isWord2InVocab and (any(c.isupper() for c in word2) or '_' in word2 or ' ' in word2):
                    split_words2 = split_by_caps_underscores_spaces(word2)
                    # 分割された単語がボキャブラリーにあるかチェック
                    if all(w in model for w in split_words2):
                        processed_word2 = split_words2
                        isWord2InVocab = True
                    
                # 分割後の単語で類似度計算
                if isWord1InVocab and isWord2InVocab:
                    try:
                        # 分割された場合、分割された単語の平均ベクトルを使用
                        if isinstance(processed_word1, list):
                            vector1 = mean([model.get_vector(w) for w in processed_word1], axis=0)
                        else:
                            vector1 = model.get_vector(processed_word1)
                        
                        if isinstance(processed_word2, list):
                            vector2 = mean([model.get_vector(w) for w in processed_word2], axis=0)
                        else:
                            vector2 = model.get_vector(processed_word2)
                        
                        # コサイン類似度でベクトル間の類似度を計算
                        similarity = calculate_similarity_between_vectors(vector1, vector2)
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
            "similarity": similarity,
            "error": errorMessage
        })

    return result
