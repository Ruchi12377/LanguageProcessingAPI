"""
SQLiteを使用したワードベクトルキャッシュのユーティリティモジュール
"""
import os
import sqlite3
import pickle
import logging
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

# ロガーの設定
logger = logging.getLogger(__name__)

class VectorCache:
    """SQLiteを使用してワードベクトルとテキスト間の類似度をキャッシュするクラス"""
    
    def __init__(self, db_path: str = None):
        """キャッシュデータベースを初期化
        
        Args:
            db_path (str, optional): データベースファイルのパス
                Noneの場合、インスタンスフォルダに作成します
        """
        # デフォルトパスの設定（Noneの場合）
        if db_path is None:
            app_root = Path(__file__).parent.parent.parent
            db_path = os.path.join(app_root, 'instance', 'vector_cache.db')
            
            # インスタンスディレクトリの存在確認
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        logger.info(f"Vector cache database path: {self.db_path}")
        
        # データベースの初期化
        self._initialize_db()
    
    def _initialize_db(self):
        """キャッシュデータベーステーブルの初期化"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # word_vectors テーブル作成
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS word_vectors (
                    word TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # similarity_cache テーブル作成
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS similarity_cache (
                    text1 TEXT NOT NULL,
                    text2 TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    similarity REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (text1, text2, model_name)
                )
                ''')
                
                conn.commit()
            logger.info("Vector cache database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector cache database: {str(e)}")
    
    def get_vector(self, word: str, model_name: str) -> Optional[np.ndarray]:
        """キャッシュからベクトルを取得
        
        Args:
            word (str): 検索する単語
            model_name (str): モデル名
            
        Returns:
            Optional[np.ndarray]: キャッシュにある場合はベクトル、なければNone
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT vector FROM word_vectors WHERE word = ? AND model_name = ?",
                    (word, model_name)
                )
                result = cursor.fetchone()
                
                if result:
                    # バイナリデータからnumpy配列に戻す
                    vector_bytes = result[0]
                    return pickle.loads(vector_bytes)
                return None
        except Exception as e:
            logger.error(f"Error retrieving vector from cache: {str(e)}")
            return None
    
    def save_vector(self, word: str, model_name: str, vector: np.ndarray) -> bool:
        """ベクトルをキャッシュに保存
        
        Args:
            word (str): 単語
            model_name (str): モデル名
            vector (np.ndarray): 保存するベクトル
            
        Returns:
            bool: 保存に成功したらTrue
        """
        try:
            # numpy配列をバイナリにシリアライズ
            vector_bytes = pickle.dumps(vector)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO word_vectors (word, model_name, vector)
                    VALUES (?, ?, ?)
                    """,
                    (word, model_name, vector_bytes)
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving vector to cache: {str(e)}")
            return False
    
    def clear_cache(self, model_name: Optional[str] = None) -> bool:
        """キャッシュをクリア
        
        Args:
            model_name (Optional[str]): 特定のモデルのキャッシュのみをクリアする場合に指定
            
        Returns:
            bool: クリアに成功したらTrue
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if model_name:
                    cursor.execute("DELETE FROM word_vectors WHERE model_name = ?", (model_name,))
                else:
                    cursor.execute("DELETE FROM word_vectors")
                conn.commit()
            logger.info(f"Cleared vector cache for model: {model_name if model_name else 'all models'}")
            return True
        except Exception as e:
            logger.error(f"Error clearing vector cache: {str(e)}")
            return False
    
    def get_stats(self) -> dict:
        """キャッシュの統計情報を取得
        
        Returns:
            dict: キャッシュの統計情報
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 全体のエントリ数
                cursor.execute("SELECT COUNT(*) FROM word_vectors")
                total_count = cursor.fetchone()[0]
                
                # モデル別のエントリ数
                cursor.execute("SELECT model_name, COUNT(*) FROM word_vectors GROUP BY model_name")
                model_counts = {row[0]: row[1] for row in cursor.fetchall()}
                
                return {
                    "total_entries": total_count,
                    "entries_by_model": model_counts
                }
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {"error": str(e)}
    
    def get_similarity(self, text1: str, text2: str, model_name: str) -> Optional[float]:
        """キャッシュから類似度を取得
        
        Args:
            text1 (str): 比較するテキスト1
            text2 (str): 比較するテキスト2
            model_name (str): モデル名
            
        Returns:
            Optional[float]: キャッシュにある場合は類似度、なければNone
        """
        try:
            # テキストの順序を正規化（text1とtext2の順序は関係ない）
            if text1 > text2:
                text1, text2 = text2, text1
                
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT similarity FROM similarity_cache WHERE text1 = ? AND text2 = ? AND model_name = ?",
                    (text1, text2, model_name)
                )
                result = cursor.fetchone()
                
                if result:
                    logger.debug(f"Similarity cache hit for '{text1[:20]}...' and '{text2[:20]}...' using model {model_name}")
                    return result[0]
                logger.debug(f"Similarity cache miss for '{text1[:20]}...' and '{text2[:20]}...' using model {model_name}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving similarity from cache: {str(e)}")
            return None
    
    def save_similarity(self, text1: str, text2: str, model_name: str, similarity: float) -> bool:
        """類似度をキャッシュに保存
        
        Args:
            text1 (str): 比較するテキスト1
            text2 (str): 比較するテキスト2
            model_name (str): モデル名
            similarity (float): 類似度
            
        Returns:
            bool: 保存に成功したらTrue
        """
        try:
            # テキストの順序を正規化（text1とtext2の順序は関係ない）
            if text1 > text2:
                text1, text2 = text2, text1
                
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO similarity_cache (text1, text2, model_name, similarity)
                    VALUES (?, ?, ?, ?)
                    """,
                    (text1, text2, model_name, similarity)
                )
                conn.commit()
            logger.debug(f"Saved similarity {similarity:.4f} for '{text1[:20]}...' and '{text2[:20]}...' using model {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error saving similarity to cache: {str(e)}")
            return False
    
    def get_or_compute_similarity(self, text1: str, text2: str, model_name: str, compute_func) -> float:
        """キャッシュから類似度を取得、なければ計算して保存
        
        Args:
            text1 (str): 比較するテキスト1
            text2 (str): 比較するテキスト2
            model_name (str): モデル名
            compute_func: キャッシュミスの場合に類似度を計算するコールバック関数
                          この関数は(text1, text2)を引数として受け取り、類似度を返す必要がある
            
        Returns:
            float: 類似度
        """
        # まずキャッシュを確認
        cached_similarity = self.get_similarity(text1, text2, model_name)
        
        if cached_similarity is not None:
            return cached_similarity
        
        # キャッシュミスの場合は計算
        logger.info(f"Computing similarity for '{text1[:20]}...' and '{text2[:20]}...' using model {model_name}")
        computed_similarity = compute_func(text1, text2)
        
        # 新しく計算した類似度をキャッシュに保存
        self.save_similarity(text1, text2, model_name, computed_similarity)
        
        return computed_similarity
    
    def clear_similarity_cache(self, model_name: Optional[str] = None) -> bool:
        """類似度キャッシュをクリア
        
        Args:
            model_name (Optional[str]): 特定のモデルのキャッシュのみをクリアする場合に指定
            
        Returns:
            bool: クリアに成功したらTrue
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if model_name:
                    cursor.execute("DELETE FROM similarity_cache WHERE model_name = ?", (model_name,))
                else:
                    cursor.execute("DELETE FROM similarity_cache")
                conn.commit()
            logger.info(f"Cleared similarity cache for model: {model_name if model_name else 'all models'}")
            return True
        except Exception as e:
            logger.error(f"Error clearing similarity cache: {str(e)}")
            return False
