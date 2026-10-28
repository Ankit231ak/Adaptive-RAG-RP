"""
Oracle labelling logic for training controller.

Computes delta-F1 = F1@10 - F1@5 and labels MORE if delta-F1 > delta_threshold, else STOP.
Note: Oracle labels are strictly for training/validation supervision; never available at test inference.
"""
from typing import Dict, List, Tuple, Any


def compute_oracle_label(f1_d5: float, f1_d10: float, delta_threshold: float = 0.05) -> Tuple[str, float]:
    """
    Compute oracle label based on marginal improvement delta F1.
    If delta_f1 > delta_threshold, MORE retrieval is justified.
    Otherwise STOP early.
    """
    delta_f1 = f1_d10 - f1_d5
    label = "MORE" if delta_f1 > delta_threshold else "STOP"
    return label, round(delta_f1, 4)


def create_oracle_record(
    question_id: str,
    question: str,
    gold_answer: str,
    ans_d5: str,
    ans_d10: str,
    f1_d5: float,
    f1_d10: float,
    delta_threshold: float = 0.05
) -> Dict[str, Any]:
    """Create a complete audit record for an oracle training example."""
    label, delta = compute_oracle_label(f1_d5, f1_d10, delta_threshold)
    return {
        "question_id": question_id,
        "question": question,
        "gold_answer": gold_answer,
        "answer_d5": ans_d5,
        "answer_d10": ans_d10,
        "f1_d5": round(f1_d5, 4),
        "f1_d10": round(f1_d10, 4),
        "delta_f1": delta,
        "oracle_label": label,
        "delta_threshold": delta_threshold
    }
