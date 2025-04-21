"""
テキスト処理のユーティリティ関数を提供するモジュール
"""
from typing import List, Dict, Any, Optional
import MeCab
import numpy as np
from numpy import ndarray, dot, mean
from numpy.linalg import norm

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

def get_word_vector(word: str, model) -> Dict[str, Any]:
    """単語のベクトルを取得する。PlamoEmbeddingモデルを使用。

    Args:
        word (str): 単語
        model: PlamoEmbeddingモデル

    Returns:
        Dict[str, Any]: 結果の辞書
    """
    from app.core.plamo_embedding import PlamoEmbedding
    if not isinstance(model, PlamoEmbedding) or not model.is_initialized:
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

def calculate_average_vector(texts: List[str], model) -> Dict[str, Any]:
    """テキストのリストから平均ベクトルを計算する。PlamoEmbeddingモデルを使用。
    
    Args:
        texts (List[str]): テキストのリスト
        model: PlamoEmbeddingモデル
        mecab_tagger (Optional[MeCab.Tagger]): MeCabのTaggerオブジェクト（現在は使用しない）
        
    Returns:
        Dict[str, Any]: 結果の辞書（成功の場合はベクトルと成功フラグ、失敗の場合はエラーメッセージ）
    """
    if not texts:
        return {"success": False, "error": "Empty text list"}
    
    # PlamoEmbeddingモデルのみサポート
    from app.core.plamo_embedding import PlamoEmbedding
    if not isinstance(model, PlamoEmbedding) or not model.is_initialized:
        return {"success": False, "error": "Plamo embedding model is not initialized"}
    
    try:
        # テキストを埋め込みベクトルに変換
        embeddings = model.encode_documents(texts)
        
        # 平均ベクトルを計算（複数テキストの場合）
        if len(embeddings) > 1:
            average_vector = mean(embeddings, axis=0)
        else:
            average_vector = embeddings[0]
        
        return {
            "success": True,
            "vector": average_vector.tolist(),  # JSONシリアライズ可能な形式に変換
            "model": "plamo-embedding"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error computing embeddings with Plamo model: {str(e)}"
        }

def process_word_pairs(pairs: List, model) -> List[Dict[str, Any]]:
    """単語ペアの処理を行い類似度を計算する
    
    Args:
        pairs (List): 単語ペアのリスト
        model: PlamoEmbeddingモデル
        
    Returns:
        List[Dict[str, str]]: 類似度計算結果のリスト
    """
    result = []
    from app.core.plamo_embedding import PlamoEmbedding
    
    if not isinstance(model, PlamoEmbedding):
        return [{"error": "Only Plamo embedding model is supported"}]
    
    for pair in pairs:
        word1 = pair[0] if pair[0] is not None else ""
        word2 = pair[1] if pair[1] is not None else ""
        if word1 == "" or word2 == "":
            continue

        similarity = 0
        errorMessage = ""

        if not model.is_initialized:
            errorMessage = "Plamo embedding model is not initialized"
        else:
            try:
                # PlamoEmbeddingの場合は直接テキスト間の類似度を計算できる
                similarity = model.similarity(word1, word2)
            except Exception as e:
                errorMessage = f"Failed to calculate similarity with Plamo model: {str(e)}"

        result.append({
            "word1": word1,
            "word2": word2,
            "similarity": similarity,
            "error": errorMessage
        })

    return result
