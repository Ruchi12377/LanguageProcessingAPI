"""
テキスト処理のユーティリティ関数を提供するモジュール
"""
from typing import List, Dict, Any, Optional
from numpy import mean
from flask import current_app
import logging

# ロガーの設定
logger = logging.getLogger(__name__)



def get_word_vector(word: str, model) -> Dict[str, Any]:
    """単語のベクトルを取得する。PlamoEmbeddingモデルを使用。キャッシュから取得できる場合はそれを使用。

    Args:
        word (str): 単語
        model: PlamoEmbeddingモデル

    Returns:
        Dict[str, Any]: 結果の辞書
    """
    from app.core.plamo_embedding import PlamoEmbedding
    from app.utils.vector_cache import VectorCache
    
    if not isinstance(model, PlamoEmbedding) or not model.is_initialized:
        return {"success": False, "error": "Plamo embedding model is not initialized"}
    
    # モデル名の取得
    model_name = getattr(model, 'model_name', 'plamo-embedding')
    
    try:
        # キャッシュを常に使用
        vector_cache = current_app.config.get('VECTOR_CACHE')
        if not vector_cache:
            vector_cache = VectorCache()
            # 新しく作成したキャッシュを設定
            current_app.config['VECTOR_CACHE'] = vector_cache
        
        # キャッシュ内にあるか確認
        cached_vector = vector_cache.get_vector(word, model_name)
        if cached_vector is not None:
            logger.debug(f"Cache hit for word: {word}")
            return {
                "success": True,
                "vector": cached_vector.tolist(),
                "model": model_name,
                "from_cache": True
            }
        
        # キャッシュになければ計算
        vector = model.get_vector(word)
        
        # 常にキャッシュに保存
        vector_cache.save_vector(word, model_name, vector)
        logger.debug(f"Cached vector for word: {word}")
        
        return {
            "success": True,
            "vector": vector.tolist(),
            "model": model_name
        }
    except Exception as e:
        logger.error(f"Error computing embedding for word '{word}': {str(e)}")
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
    
    # モデル名の取得
    model_name = getattr(model, 'model_name', 'plamo-embedding')
    
    try:
        # キャッシュを常に使用
        from app.utils.vector_cache import VectorCache
        
        vector_cache = current_app.config.get('VECTOR_CACHE')
        if not vector_cache:
            vector_cache = VectorCache()
            # 新しく作成したキャッシュを設定
            current_app.config['VECTOR_CACHE'] = vector_cache
        
        # 全テキストに対する埋め込みベクトル（キャッシュ利用または新規計算）
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # まず、キャッシュからベクトルを取得
        for i, text in enumerate(texts):
            cached_vector = vector_cache.get_vector(text, model_name)
            if cached_vector is not None:
                embeddings.append(cached_vector)
                logger.debug(f"Cache hit for text: {text[:20]}...")
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # キャッシュにないテキストはモデルで計算
        if uncached_texts:
            new_embeddings = model.encode_query(uncached_texts)
            
            # 常にキャッシュに保存
            for i, text in enumerate(uncached_texts):
                vector_cache.save_vector(text, model_name, new_embeddings[i])
                logger.debug(f"Cached vector for text: {text[:20]}...")
            
            # 正しい位置に挿入するため、元の順序を維持
            full_embeddings = [None] * len(texts)
            cached_idx = 0
            
            for i in range(len(texts)):
                if i in uncached_indices:
                    # キャッシュになかったテキスト
                    unc_idx = uncached_indices.index(i)
                    full_embeddings[i] = new_embeddings[unc_idx]
                else:
                    # キャッシュにあったテキスト
                    full_embeddings[i] = embeddings[cached_idx]
                    cached_idx += 1
                    
            embeddings = full_embeddings
        
        # 平均ベクトルを計算（複数テキストの場合）
        if len(embeddings) > 1:
            average_vector = mean(embeddings, axis=0)
        else:
            average_vector = embeddings[0]
        
        return {
            "success": True,
            "vector": average_vector.tolist(),  # JSONシリアライズ可能な形式に変換
            "model": model_name
        }
    except Exception as e:
        logger.error(f"Error computing embeddings: {str(e)}")
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
