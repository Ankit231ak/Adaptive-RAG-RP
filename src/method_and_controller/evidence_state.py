"""
V1 Post-Retrieval Evidence State Features.

Implements the seven V1 evidence-state features extracted strictly after initial top-5 retrieval:
1. Top-1 score: s1
2. Mean score: mean(s1..s5)
3. Score std: std(s1..s5)
4. Top-1 to Top-2 margin: s1 - s2
5. Score range: s1 - s5
6. Max similarity: max(q, d_i)
7. Mean pairwise redundancy: mean cosine similarity among doc embeddings

STRICT LEAKAGE GUARDRAIL:
No gold answers, no gold supporting facts, no gold document IDs, and no future answer F1
are allowed as features. Features must be computable purely from the query and top-5 retrieved evidence.
"""
from dataclasses import dataclass, asdict
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.data_and_retrieval.corpus import Document
from src.data_and_retrieval.embeddings import l2_normalize

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "top1_score",
    "mean_score",
    "score_std",
    "margin_top1_top2",
    "score_range",
    "max_similarity",
    "mean_redundancy"
]


@dataclass
class EvidenceFeatures:
    """Dataclass holding the 7 V1 evidence-state features."""
    top1_score: float
    mean_score: float
    score_std: float
    margin_top1_top2: float
    score_range: float
    max_similarity: float
    mean_redundancy: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def to_vector(self) -> List[float]:
        return [
            self.top1_score,
            self.mean_score,
            self.score_std,
            self.margin_top1_top2,
            self.score_range,
            self.max_similarity,
            self.mean_redundancy
        ]


def compute_pairwise_redundancy(doc_embeddings: np.ndarray) -> float:
    """
    Compute mean pairwise cosine similarity across retrieved document embeddings.
    Formula: mean_{i < j} cosine(d_i, d_j)
    """
    n = len(doc_embeddings)
    if n <= 1:
        return 0.0

    # Ensure L2 normalization
    norm_embs = l2_normalize(doc_embeddings)
    # Cosine similarity matrix
    sim_matrix = np.dot(norm_embs, norm_embs.T)

    # Extract upper triangle excluding diagonal
    upper_tri_indices = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[upper_tri_indices]

    if len(pairwise_sims) == 0:
        return 0.0

    return float(np.mean(pairwise_sims))


def extract_evidence_features(
    scores: List[float],
    doc_embeddings: Optional[np.ndarray] = None,
    query_embedding: Optional[np.ndarray] = None,
    default_max_sim: Optional[float] = None
) -> EvidenceFeatures:
    """
    Extract 7 V1 evidence-state features from top-5 retrieved documents.
    Accepts retrieval similarity scores and optional dense embeddings.
    """
    if not scores:
        return EvidenceFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    scores_arr = np.array(scores, dtype=np.float32)
    s1 = float(scores_arr[0])
    s2 = float(scores_arr[1]) if len(scores_arr) > 1 else s1
    s5 = float(scores_arr[-1])

    mean_sc = float(np.mean(scores_arr))
    std_sc = float(np.std(scores_arr))
    margin = float(s1 - s2)
    sc_range = float(s1 - s5)

    # Max similarity to query
    if query_embedding is not None and doc_embeddings is not None and len(doc_embeddings) > 0:
        q_norm = l2_normalize(query_embedding.reshape(1, -1))
        d_norm = l2_normalize(doc_embeddings)
        sims = np.dot(d_norm, q_norm.T).squeeze()
        max_sim = float(np.max(sims))
    elif default_max_sim is not None:
        max_sim = float(default_max_sim)
    else:
        max_sim = s1  # Since retrieval scores in FAISS are inner-product similarities

    # Mean pairwise redundancy
    if doc_embeddings is not None and len(doc_embeddings) > 1:
        mean_red = compute_pairwise_redundancy(doc_embeddings)
    else:
        # Heuristic fallback if embeddings not directly attached
        mean_red = max(0.0, min(1.0, mean_sc * 0.8))

    return EvidenceFeatures(
        top1_score=round(s1, 4),
        mean_score=round(mean_sc, 4),
        score_std=round(std_sc, 4),
        margin_top1_top2=round(margin, 4),
        score_range=round(sc_range, 4),
        max_similarity=round(max_sim, 4),
        mean_redundancy=round(mean_red, 4)
    )


def validate_evidence_features(features: EvidenceFeatures) -> Dict[str, bool]:
    """
    Sanity validation check for Day 6 milestone:
    Confirm features are numerically sensible, non-NaN, and within mathematical bounds.
    """
    v = features.to_vector()
    checks = {
        "no_nan": not any(np.isnan(x) for x in v),
        "no_inf": not any(np.isinf(x) for x in v),
        "margin_non_negative": features.margin_top1_top2 >= -1e-6,
        "score_range_non_negative": features.score_range >= -1e-6,
        "redundancy_in_bounds": -1.0 <= features.mean_redundancy <= 1.0,
        "std_non_negative": features.score_std >= 0.0,
    }
    return checks
