"""
単語ベクトル処理のコア機能を提供するモジュール
"""
from typing import List, Dict, Any
import os
import gensim
import MeCab
from numpy import mean, ndarray

from app.utils.text_processing import (
    normalize_vector, calculate_vector_for_word,
    extract_words_from_text, split_by_caps_underscores_spaces
)

class VectorModel:
    """単語ベクトルモデルのラッパークラス"""
    
    def __init__(self, model_path: str = "./model_gensim_norm"):
        """モデルを初期化する
        
        Args:
            model_path (str): モデルファイルのパス
        
        Raises:
            FileNotFoundError: モデルファイルが見つからない場合
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = gensim.models.KeyedVectors.load(model_path, mmap='r')
        
    def __contains__(self, word: str) -> bool:
        """単語がモデルに存在するかチェックする
        
        Args:
            word (str): チェックする単語
            
        Returns:
            bool: 単語がモデルに存在する場合はTrue、そうでない場合はFalse
        """
        return word in self.model
        
    def get_vector(self, word: str) -> ndarray:
        """単語のベクトルを取得する
        
        Args:
            word (str): 単語
            
        Returns:
            ndarray: 単語のベクトル
            
        Raises:
            KeyError: 単語がモデルに存在しない場合
        """
        return self.model[word]
    
    def similarity(self, word1: str, word2: str) -> float:
        """2つの単語間の類似度を計算する
        
        Args:
            word1 (str): 1つ目の単語
            word2 (str): 2つ目の単語
            
        Returns:
            float: 類似度スコア
            
        Raises:
            KeyError: どちらかの単語がモデルに存在しない場合
        """
        return float(self.model.similarity(word1, word2))

def get_word_vector(word: str, model: VectorModel, mecab_tagger: MeCab.Tagger) -> Dict[str, Any]:
    """単語のベクトルを取得する。単語がモデルに存在しない場合は分かち書きして名詞や形容詞を取り出し平均ベクトルを返す
    
    Args:
        word (str): 単語
        model (VectorModel): 単語ベクトルモデル
        mecab_tagger (MeCab.Tagger): MeCabのTaggerオブジェクト
        
    Returns:
        Dict[str, Any]: 結果の辞書
    """
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

def calculate_average_vector(texts: List[str], model: VectorModel, mecab_tagger: MeCab.Tagger) -> Dict[str, Any]:
    """テキストのリストから平均ベクトルを計算する。日本語テキストの場合は分かち書きで処理する
    
    Args:
        texts (List[str]): テキストのリスト
        model (VectorModel): 単語ベクトルモデル
        mecab_tagger (MeCab.Tagger): MeCabのTaggerオブジェクト
        
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
        # テキストから名詞、形容詞、動詞を抽出
        extracted = extract_words_from_text(text, mecab_tagger)
        words_to_process = [word_info["word"] for word_info in extracted["ordered_words"]]
        
        # 抽出された単語がある場合
        if words_to_process:
            # モデルに存在する単語だけを使用
            valid_words = [w for w in words_to_process if w in model]
            
            if valid_words:
                # 単語のベクトルの平均を計算
                text_vectors = [model.get_vector(w) for w in valid_words]
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

def process_word_pairs(pairs: List, model: VectorModel) -> List[Dict[str, Any]]:
    """単語ペアの処理を行い類似度を計算する
    
    Args:
        pairs (List): 単語ペアのリスト
        model (VectorModel): 単語ベクトルモデル
        
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
                    from numpy.linalg import norm
                    from numpy import dot
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
            "similarity": similarity,
            "error": errorMessage
        })

    return result
