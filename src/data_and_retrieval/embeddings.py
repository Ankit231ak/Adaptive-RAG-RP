"""
Dense Embedding Interface supporting local SentenceTransformers and Ollama Embeddings.
"""
from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Optional, Any
import numpy as np
import requests

logger = logging.getLogger(__name__)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors along axis 1 to unit L2 norm."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return vectors / norms


class BaseEmbeddingModel(ABC):
    """Abstract Base Class for Dense Embedding Models."""
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of text strings into normalized 2D numpy array of shape (N, D)."""
        pass

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query into a 1D numpy array of shape (D,)."""
        vectors = self.embed_texts([query])
        return vectors[0]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass


class SentenceTransformerEmbeddingModel(BaseEmbeddingModel):
    """Local SentenceTransformer embedding model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        if hasattr(self._model, "get_embedding_dimension"):
            self._dimension = self._model.get_embedding_dimension()
        else:
            self._dimension = self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return l2_normalize(np.asarray(embeddings, dtype=np.float32))

    @property
    def dimension(self) -> int:
        return self._dimension


class OllamaEmbeddingModel(BaseEmbeddingModel):
    """
    Ollama Embedding Model.
    Connects to local Ollama API (http://localhost:11434).
    """


    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "",  # Space for user to provide Ollama model name
        dimension: int = 384,
        timeout: int = 30,
        fallback_model: Optional[BaseEmbeddingModel] = None
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name.strip()
        self._dimension = dimension
        self.timeout = timeout
        self._fallback = fallback_model

    @property
    def fallback(self) -> BaseEmbeddingModel:
        if self._fallback is None:
            self._fallback = SentenceTransformerEmbeddingModel()
        return self._fallback

    @property
    def dimension(self) -> int:
        return self._dimension

    def is_available(self) -> bool:
        if not self.model_name:
            return False
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        if not self.model_name:
            # Space left for user; gracefully use fallback
            return self.fallback.embed_texts(texts)

        try:
            from tqdm import tqdm
            import time
            # Try new Ollama embed API (/api/embed) in chunks of 100
            chunk_size = 10
            
            all_embeddings = []

            if len(texts) > 1:
                print(f"\nSending {len(texts)} documents to Ollama for embedding in chunks of {chunk_size}...")
                iterator = tqdm(range(0, len(texts), chunk_size), desc="Ollama Embedding")
            else:
                iterator = range(0, len(texts), chunk_size)
            for i in iterator:
                chunk = texts[i:i + chunk_size]
                res = requests.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model_name, "input": chunk},
                    timeout=self.timeout
                )
                if res.status_code == 200:
                    data = res.json()
                    chunk_embs = data.get("embeddings", [])
                    if len(chunk_embs) != len(chunk):
                        raise RuntimeError("Mismatch in returned embedding lengths")
                    all_embeddings.extend(chunk_embs)
                else:
                    raise RuntimeError(f"Ollama API returned HTTP {res.status_code}: {res.text}")
                time.sleep(0.5) # Give local Ollama a split second to breathe
                    
            if len(all_embeddings) == len(texts):
                return l2_normalize(np.array(all_embeddings, dtype=np.float32))

            # Try legacy Ollama embeddings API (/api/embeddings) per item
            results = []
            for t in texts:
                r = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model_name, "prompt": t},
                    timeout=self.timeout
                )
                if r.status_code == 200:
                    results.append(r.json().get("embedding", []))
                else:
                    raise RuntimeError(f"Ollama API returned HTTP {r.status_code}: {r.text}")

            vecs = np.array(results, dtype=np.float32)
            return l2_normalize(vecs)
        except Exception as e:
            logger.warning(
                f"Ollama embedding call failed for model '{self.model_name}': {e}. "
                "Using local SentenceTransformer fallback."
            )
            return self.fallback.embed_texts(texts)


def get_embedding_model(config: Optional[Dict[str, Any]] = None) -> BaseEmbeddingModel:
    """Factory function to instantiate embedding model based on config."""
    cfg = config or {}
    retrieval_cfg = cfg.get("retrieval", {})
    provider = retrieval_cfg.get("embedding_provider", "sentence_transformers").lower()

    if provider == "ollama":
        ollama_cfg = retrieval_cfg.get("ollama", {})
        model_name = ollama_cfg.get("model_name", "")
        base_url = ollama_cfg.get("base_url", "http://localhost:11434")
        timeout = ollama_cfg.get("timeout", 30)
        dimension = retrieval_cfg.get("dimension", 384)
        return OllamaEmbeddingModel(base_url=base_url, model_name=model_name, timeout=timeout, dimension=dimension)

    model_name = retrieval_cfg.get("sentence_transformer_model", "all-MiniLM-L6-v2")
    return SentenceTransformerEmbeddingModel(model_name=model_name)
