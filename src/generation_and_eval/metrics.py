"""
Evaluation metrics: Answer quality (EM/F1) and Retrieval accuracy (Recall@k, MRR).
"""
from collections import Counter
import re
import string
from typing import Dict, List, Optional, Tuple, Any
from src.data_and_retrieval.corpus import Document


def normalize_answer(s: str) -> str:
    """Standard HotpotQA answer normalizer."""
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Official Exact Match (EM) computation."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """Official token-level F1, Precision, and Recall."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRICS = (0.0, 0.0, 0.0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRICS
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRICS

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()

    if not prediction_tokens or not ground_truth_tokens:
        is_exact = (prediction_tokens == ground_truth_tokens)
        score = 1.0 if is_exact else 0.0
        return (score, score, score)

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return ZERO_METRICS

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return (f1, precision, recall)


def evaluate_predictions(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """Aggregate answer quality metrics over dataset."""
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Mismatch in length: {len(predictions)} predictions vs {len(ground_truths)} ground truths.")

    total = len(predictions)
    if total == 0:
        return {"exact_match": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "total_samples": 0}

    em_total = 0.0
    f1_total = 0.0
    prec_total = 0.0
    rec_total = 0.0

    for pred, gt in zip(predictions, ground_truths):
        em_total += 1.0 if exact_match_score(pred, gt) else 0.0
        f1, prec, rec = f1_score(pred, gt)
        f1_total += f1
        prec_total += prec
        rec_total += rec

    return {
        "exact_match": round(em_total / total, 4),
        "f1": round(f1_total / total, 4),
        "precision": round(prec_total / total, 4),
        "recall": round(rec_total / total, 4),
        "total_samples": total
    }


# =========================================================================
# Retrieval Metrics (Days 5 & 6 Milestone)
# =========================================================================

def recall_at_k(retrieved_doc_ids: List[str], gold_doc_ids: List[str], k: int) -> float:
    """
    Fraction of gold documents retrieved within top-k documents.
    """
    if not gold_doc_ids:
        return 1.0
    top_k_ids = set(retrieved_doc_ids[:k])
    hits = sum(1 for gid in gold_doc_ids if gid in top_k_ids)
    return hits / len(gold_doc_ids)


def mrr_at_k(retrieved_doc_ids: List[str], gold_doc_ids: List[str], k: int) -> float:
    """
    Mean Reciprocal Rank (MRR) of the first retrieved gold document in top-k.
    """
    if not gold_doc_ids:
        return 1.0
    gold_set = set(gold_doc_ids)
    for rank, did in enumerate(retrieved_doc_ids[:k], start=1):
        if did in gold_set:
            return 1.0 / rank
    return 0.0


def supporting_fact_coverage(
    retrieved_docs: List[Document],
    supporting_facts: List[Any]
) -> float:
    """
    Fraction of gold supporting fact titles found in the retrieved documents.
    """
    if not supporting_facts:
        return 1.0
    retrieved_titles = set(d.title for d in retrieved_docs)
    sf_titles = set(
        sf.title if hasattr(sf, "title") else sf[0]
        for sf in supporting_facts
    )
    if not sf_titles:
        return 1.0
    matched = sum(1 for t in sf_titles if t in retrieved_titles)
    return matched / len(sf_titles)


def evaluate_retrieval(
    retrieval_records: List[Dict[str, Any]],
    k_values: List[int] = [3, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate retrieval quality across multiple queries for given k values.
    Each record must have 'retrieved_doc_ids' and 'gold_doc_ids'.
    """
    if not retrieval_records:
        return {}

    summary = {}
    for k in k_values:
        recalls = [
            recall_at_k(r["retrieved_doc_ids"], r["gold_doc_ids"], k)
            for r in retrieval_records
        ]
        mrrs = [
            mrr_at_k(r["retrieved_doc_ids"], r["gold_doc_ids"], k)
            for r in retrieval_records
        ]
        summary[f"recall_at_{k}"] = round(float(sum(recalls) / len(recalls)), 4)
        summary[f"mrr_at_{k}"] = round(float(sum(mrrs) / len(mrrs)), 4)

    summary["total_queries"] = len(retrieval_records)
    return summary
