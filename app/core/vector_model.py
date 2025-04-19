"""
単語ベクトル処理のコア機能を提供するモジュール
"""
from typing import List, Dict, Any
import os
import MeCab
from numpy import mean, ndarray
import numpy as np

from app.utils.text_processing import (
    normalize_vector, calculate_vector_for_word,
    extract_words_from_text, split_by_caps_underscores_spaces
)

# PlamoEmbedding以外のモデルが必要な場合のためにGensimモデル機能を保持
class VectorModel:
    """単語ベクトルモデルのラッパークラス (Plamo使用時はこのクラスは使用されない)"""
    
    def __init__(self, model_path: str = "./model_gensim_norm"):
        """モデルを初期化する
        
        Args:
            model_path (str): モデルファイルのパス
        
        Raises:
            FileNotFoundError: モデルファイルが見つからない場合
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        # 遅延インポートでGensimに依存しないようにする
        import gensim
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

def get_word_vector(word: str, model, mecab_tagger: MeCab.Tagger) -> Dict[str, Any]:
    """単語のベクトルを取得する。モデルに応じた適切な処理を行う
    
    Args:
        word (str): 単語
        model: ベクトルモデル（PlamoEmbeddingまたはVectorModel）
        mecab_tagger (MeCab.Tagger): MeCabのTaggerオブジェクト
        
    Returns:
        Dict[str, Any]: 結果の辞書
    """
    # PlamoEmbeddingモデルの場合
    from app.core.plamo_embedding import PlamoEmbedding
    if isinstance(model, PlamoEmbedding):
        if not model.is_initialized:
            return {"success": False, "error": "Plamo embedding model is not initialized"}
        try:
            # Plamoの場合はどんな単語も直接ベクトル化できる
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

def calculate_average_vector(texts: List[str], model, mecab_tagger: MeCab.Tagger = None) -> Dict[str, Any]:
    """テキストのリストから平均ベクトルを計算する。
    PlamoEmbeddingモデルが利用可能な場合はそれを使用し、
    そうでない場合は従来の分かち書きと単語ベクトルの平均を計算する。
    
    Args:
        texts (List[str]): テキストのリスト
        model: ベクトルモデル（VectorModelまたはPlamoEmbedding）
        mecab_tagger (MeCab.Tagger, optional): MeCabのTaggerオブジェクト（従来モデルの場合のみ使用）
        
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
    
    # 従来のVectorModelの場合は元の実装を使用
    # 各テキストのベクトルを計算
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
