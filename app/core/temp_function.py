from typing import List

import MeCab
from numpy import mean
from app.core.vector_model import VectorModel
from app.utils.text_processing import calculate_vector_for_word, normalize_vector
from utils import extract_words_from_text


def calculate_average_vector(texts: List[str], model: VectorModel, mecab_tagger: MeCab.Tagger) -> Dict[str, Any]:
    """テキストのリストから平均ベクトルを計算する。日本語テキストの場合は分かち書きで処理する。
    空白およびアンダースコアで区切られたテキストは複合語として処理し、平均ベクトルを返す。
    
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
    }
    
    # 一部の単語が見つからなかった場合は警告を追加
    if missing_words:
        result["warning"] = f"Some words not found in model vocabulary: {', '.join(missing_words)}"
    
    # 処理された単語情報を追加
    if processed_words:
        result["processed_words"] = processed_words
        
    return result
