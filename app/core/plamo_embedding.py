"""
Plamo Embedding モデルを使用してテキスト埋め込みを行うモジュール
"""
import logging
from typing import List, Optional
import numpy as np

# ロガーの設定
logger = logging.getLogger(__name__)

class PlamoEmbedding:
    """Plamo Embedding モデルを使用してテキスト埋め込みを行うクラス"""
    
    def __init__(self, model_name: str = "pfnet/plamo-embedding-1b", cache_dir: Optional[str] = None, use_fp16: bool = False):
        """モデルを初期化する
        
        Args:
            model_name (str): Hugging Face モデル名
            cache_dir (Optional[str]): モデルキャッシュディレクトリ
            use_fp16 (bool): 半精度浮動小数点 (FP16) を使用するかどうか
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.torch = None
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        self.is_initialized = False
        self.use_fp16 = use_fp16
        
        # 必要なモジュールをインポート (インストールされていない場合に全体が失敗しないよう遅延インポート)
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            
            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Initializing Plamo embedding model '{model_name}' on {self.device}")
            
            kwargs = {"trust_remote_code": True}
            if cache_dir:
                kwargs["cache_dir"] = cache_dir
                
            # モデルロードの進捗をより詳細にロギング
            logger.info(f"Loading tokenizer from '{model_name}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
            
            logger.info(f"Loading model from '{model_name}'...")
            self.model = AutoModel.from_pretrained(model_name, low_cpu_mem_usage=True, **kwargs)
            self.model = self.model.to(self.device)
            
            # GPUメモリを節約するためのオプション設定
            if self.use_fp16 and self.torch.cuda.is_available():
                logger.info("Using FP16 precision to optimize memory usage")
                self.model = self.model.half()  # FP16に変換
            
            self.is_initialized = True
            logger.info(f"Successfully initialized Plamo embedding model '{model_name}'")
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}. Install with: pip install torch transformers")
            self.is_initialized = False
        except Exception as e:
            logger.error(f"Failed to initialize Plamo embedding model: {str(e)}")
            self.is_initialized = False
    
    def encode_query(self, text: List[str]) -> np.ndarray:
        """クエリテキストをベクトルに変換する
        
        Args:
            text (List[str]): エンコードするテキストのリスト
            
        Returns:
            np.ndarray: テキストの埋め込みベクトル
            
        Raises:
            RuntimeError: モデルが初期化されていない場合
        """
        if not self.is_initialized:
            raise RuntimeError("Plamo embedding model is not initialized")
        
        try:
            with self.torch.inference_mode():
                embedding = self.model.encode_query(text, self.tokenizer)
            
            # NumPy配列に変換して返す
            return embedding.cpu().numpy()
        except Exception as e:
            logger.error(f"Error encoding query: {str(e)}")
            raise RuntimeError(f"Failed to encode text: {str(e)}")
    
    def similarity(self, text1: str, text2: str) -> float:
        """2つのテキスト間の類似度を計算する
        
        Args:
            text1 (str): 1つ目のテキスト
            text2 (str): 2つ目のテキスト
            
        Returns:
            float: コサイン類似度
            
        Raises:
            RuntimeError: モデルが初期化されていない場合
        """
        if not self.is_initialized:
            raise RuntimeError("Plamo embedding model is not initialized")
            
        with self.torch.inference_mode():
            embedding1 = self.model.encode_document([text1], self.tokenizer)
            embedding2 = self.model.encode_document([text2], self.tokenizer)
            
            similarity = self.torch.nn.functional.cosine_similarity(embedding1, embedding2)
            
        return float(similarity.item())
        
    def get_vector(self, text: str) -> np.ndarray:
        """テキストをベクトルに変換する (VectorModelとインターフェースを合わせる)
        
        Args:
            text (str): エンコードするテキスト
            
        Returns:
            np.ndarray: テキストの埋め込みベクトル
            
        Raises:
            RuntimeError: モデルが初期化されていない場合
        """
        if not self.is_initialized:
            raise RuntimeError("Plamo embedding model is not initialized")
        try:
            # encode_queryはリストを期待するので、単一テキストをリストに変換
            return self.encode_query([text])[0]
        except Exception as e:
            logger.error(f"Error getting vector: {str(e)}")
            raise RuntimeError(f"Failed to get vector: {str(e)}")
    
    def __contains__(self, text: str) -> bool:
        """テキストがモデルで処理可能かどうかを返す
        Plamoモデルは基本的にどんなテキストも処理できるため、初期化済みであればTrueを返す
        
        Args:
            text (str): チェックするテキスト
            
        Returns:
            bool: モデルが初期化されていればTrue、そうでなければFalse
        """
        return self.is_initialized
