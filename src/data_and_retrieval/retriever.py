"""
FAISS Vector Store and Dense Retriever Implementation.
"""
from dataclasses import dataclass
import logging
import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.data_and_retrieval.corpus import Corpus, Document
from src.data_and_retrieval.embeddings import BaseEmbeddingModel, get_embedding_model, l2_normalize

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False


@dataclass
class RetrievedDoc:
    """Represents a retrieved document passage with rank and similarity score."""
    doc: Document
    score: float
    rank: int
    doc_id: str
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.doc.title,
            "score": round(self.score, 4),
            "rank": self.rank,
            "text_snippet": self.doc.text[:120] + "..." if len(self.doc.text) > 120 else self.doc.text
        }


class FAISSVectorStore:
    """
    Dense Vector Store backed by FAISS IndexFlatIP (Inner Product on normalized vectors).
    Provides seamless vectorized NumPy fallback if faiss-cpu is not installed in the environment.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.doc_ids: List[str] = []
        self.doc_id_to_idx: Dict[str, int] = {}
        self.vectors: Optional[np.ndarray] = None
        self._faiss_index = None

        if FAISS_AVAILABLE:
            self._faiss_index = faiss.IndexFlatIP(dimension)
            logger.info("Initialized FAISS IndexFlatIP.")
        else:
            logger.info("FAISS library not installed. Using pure NumPy exact cosine similarity engine.")

    def add(self, vectors: np.ndarray, doc_ids: List[str]):
        """Add normalized vectors with associated document IDs."""
        if len(vectors) == 0:
            return

        norm_vectors = l2_normalize(np.asarray(vectors, dtype=np.float32))
        start_idx = len(self.doc_ids)

        for i, did in enumerate(doc_ids):
            self.doc_ids.append(did)
            self.doc_id_to_idx[did] = start_idx + i

        if self.vectors is None:
            self.vectors = norm_vectors
        else:
            self.vectors = np.vstack([self.vectors, norm_vectors])

        if FAISS_AVAILABLE and self._faiss_index is not None:
            self._faiss_index.add(norm_vectors)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        candidate_doc_ids: Optional[List[str]] = None
    ) -> List[Tuple[str, float, int]]:
        """
        Search for top_k nearest neighbors.
        Returns list of tuples: (doc_id, similarity_score, vector_index)
        """
        if self.vectors is None or len(self.doc_ids) == 0:
            return []

        q_vec = np.asarray(query_vector, dtype=np.float32)
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)
        q_vec = l2_normalize(q_vec)

        # Case 1: Candidate filtering specified (e.g. distractor set of 10 docs)
        if candidate_doc_ids is not None:
            valid_indices = [
                self.doc_id_to_idx[did]
                for did in candidate_doc_ids
                if did in self.doc_id_to_idx
            ]
            if not valid_indices:
                return []

            sub_vectors = self.vectors[valid_indices]
            sub_scores = np.dot(sub_vectors, q_vec.T).squeeze(axis=1)
            sorted_order = np.argsort(-sub_scores)

            results = []
            for rank_idx in sorted_order[:top_k]:
                orig_idx = valid_indices[rank_idx]
                results.append((self.doc_ids[orig_idx], float(sub_scores[rank_idx]), orig_idx))
            return results

        # Case 2: Full index search via FAISS
        if FAISS_AVAILABLE and self._faiss_index is not None:
            scores, indices = self._faiss_index.search(q_vec, min(top_k, len(self.doc_ids)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.doc_ids):
                    results.append((self.doc_ids[idx], float(score), int(idx)))
            return results

        # Case 3: Full index search via NumPy
        scores = np.dot(self.vectors, q_vec.T).squeeze(axis=1)
        top_indices = np.argsort(-scores)[:top_k]
        results = []
        for idx in top_indices:
            results.append((self.doc_ids[idx], float(scores[idx]), int(idx)))
        return results

    def save(self, filepath: str):
        """Save vector store to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if FAISS_AVAILABLE and self._faiss_index is not None:
            faiss.write_index(self._faiss_index, filepath)
            np.savez(f"{filepath}.meta.npz", doc_ids=np.array(self.doc_ids), vectors=self.vectors)
        else:
            np.savez(f"{filepath}.npz", doc_ids=np.array(self.doc_ids), vectors=self.vectors)

    def load(self, filepath: str):
        """Load vector store from disk."""
        if FAISS_AVAILABLE and os.path.exists(filepath):
            self._faiss_index = faiss.read_index(filepath)
            meta_path = f"{filepath}.meta.npz"
            if os.path.exists(meta_path):
                data = np.load(meta_path)
                self.doc_ids = list(data["doc_ids"])
                self.vectors = data["vectors"]
                self.doc_id_to_idx = {did: i for i, did in enumerate(self.doc_ids)}
        elif os.path.exists(f"{filepath}.npz"):
            data = np.load(f"{filepath}.npz")
            self.doc_ids = list(data["doc_ids"])
            self.vectors = data["vectors"]
            self.doc_id_to_idx = {did: i for i, did in enumerate(self.doc_ids)}
            if FAISS_AVAILABLE:
                self._faiss_index = faiss.IndexFlatIP(self.dimension)
                self._faiss_index.add(self.vectors)


class DenseRetriever:
    """
    Retriever class serving dense ranked document passages for queries.
    Supports FAISS vector store, stable document IDs, and fixed k (3, 5, 10).
    """

    def __init__(
        self,
        corpus: Corpus,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.corpus = corpus
        self.config = config or {}
        self.embedding_model = embedding_model or get_embedding_model(self.config)
        self.vector_store = FAISSVectorStore(dimension=self.embedding_model.dimension)
        self.is_indexed = False

    def build_index(self):
        """Index all documents in corpus."""
        docs = list(self.corpus.documents.values())
        if not docs:
            return

        texts = [f"{d.title}\n{d.text}" for d in docs]
        doc_ids = [d.doc_id for d in docs]

        logger.info(f"Encoding {len(texts)} corpus passages...")
        embeddings = self.embedding_model.embed_texts(texts)
        self.vector_store.add(embeddings, doc_ids)
        self.is_indexed = True
        logger.info(f"✔ Indexed {len(doc_ids)} documents in vector store.")

    def ensure_indexed(self):
        if not self.is_indexed:
            self.build_index()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_doc_ids: Optional[List[str]] = None
    ) -> List[RetrievedDoc]:
        """
        Retrieve top_k ranked documents with scores for a given query.
        """
        self.ensure_indexed()

        query_vec = self.embedding_model.embed_query(query)
        search_results = self.vector_store.search(
            query_vector=query_vec,
            top_k=top_k,
            candidate_doc_ids=candidate_doc_ids
        )

        retrieved = []
        for rank, (doc_id, score, vec_idx) in enumerate(search_results, start=1):
            doc = self.corpus.get_document(doc_id)
            if doc is not None:
                emb = self.vector_store.vectors[vec_idx] if self.vector_store.vectors is not None else None
                retrieved.append(RetrievedDoc(
                    doc=doc,
                    score=float(score),
                    rank=rank,
                    doc_id=doc_id,
                    embedding=emb
                ))

        return retrieved

    def retrieve_split_5_and_5(
        self,
        query: str,
        candidate_doc_ids: Optional[List[str]] = None
    ) -> Tuple[List[RetrievedDoc], List[RetrievedDoc]]:
        """
        Retrieve top-10 and cleanly split into initial D5 and additional D6:10.
        Guarantees no duplicate document IDs across splits.
        """
        top10 = self.retrieve(query=query, top_k=10, candidate_doc_ids=candidate_doc_ids)
        d5 = top10[:5]
        d6_10 = top10[5:10]

        # Verify no duplicate IDs
        seen_ids = set(r.doc_id for r in d5)
        filtered_d6_10 = [r for r in d6_10 if r.doc_id not in seen_ids]

        return d5, filtered_d6_10


# Backward compatibility alias
class Retriever(DenseRetriever):
    """
    Retriever class supporting both legacy slicing and dense FAISS retrieval.
    """
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        context_doc_ids: Optional[List[str]] = None,
        candidate_doc_ids: Optional[List[str]] = None
    ) -> List[Document]:
        cands = candidate_doc_ids or context_doc_ids
        retrieved_docs = super().retrieve(query=query, top_k=top_k, candidate_doc_ids=cands)
        return [rd.doc for rd in retrieved_docs]

    def retrieve_ranked(
        self,
        query: str,
        top_k: int = 5,
        candidate_doc_ids: Optional[List[str]] = None
    ) -> List[RetrievedDoc]:
        return super().retrieve(query=query, top_k=top_k, candidate_doc_ids=candidate_doc_ids)
