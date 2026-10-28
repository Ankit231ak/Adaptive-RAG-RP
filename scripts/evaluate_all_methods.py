"""
Comprehensive Evaluation of All Baselines (B0 to B5) and Proposed Adaptive-RAG (B6).
Runs deterministically across the 1,000 precomputed queries and evaluates on the frozen Test Split.
Zero external LLM calls needed because baseline responses and 7 features are precomputed and cached.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.generation_and_eval.metrics import evaluate_predictions
from src.common.cache_builder import ensure_baseline_caches


def main():
    print("=========================================================================")
    print("       ADAPTIVE-RAG: MASTER EVALUATION OF ALL METHODS (B0 - B6)          ")
    print("=========================================================================\n")

    raw_dir = "results/raw"
    tables_dir = "results/tables"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Ensure raw caches exist, reconstructing them if missing
    ensure_baseline_caches(raw_dir=raw_dir, tables_dir=tables_dir, splits_dir="data/splits")

    # 1. Load Splits
    with open("data/splits/train_ids.json") as f:
        train_ids = json.load(f)
    with open("data/splits/val_ids.json") as f:
        val_ids = json.load(f)
    with open("data/splits/test_ids.json") as f:
        test_ids = json.load(f)

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    print(f"Loaded Splits: {len(train_ids)} Train | {len(val_ids)} Val | {len(test_ids)} Test")

    # 2. Load cached outputs
    print("\nLoading cached outputs for B0, B1, B2, B3, B4...")
    with open(os.path.join(raw_dir, "b0_direct_llm_outputs.json")) as f:
        b0_map = {x["question_id"]: x for x in json.load(f)}
    with open(os.path.join(raw_dir, "b1_fixed_rag_3_outputs.json")) as f:
        b1_map = {x["question_id"]: x for x in json.load(f)}
    with open(os.path.join(raw_dir, "b2_fixed_rag_5_outputs.json")) as f:
        b2_map = {x["question_id"]: x for x in json.load(f)}
    with open(os.path.join(raw_dir, "b3_fixed_rag_10_outputs.json")) as f:
        b3_map = {x["question_id"]: x for x in json.load(f)}
    with open(os.path.join(raw_dir, "b4_simple_score_rule_outputs.json")) as f:
        b4_map = {x["question_id"]: x for x in json.load(f)}

    # 3. Load Oracle Labels
    with open(os.path.join(raw_dir, "oracle_labels.json")) as f:
        oracle_map = {x["question_id"]: (1 if x["oracle_label"] == "MORE" else 0) for x in json.load(f)}

    # 4. Load 7 Evidence Features
    print("Loading 7 V1 Evidence Features...")
    features_map = {}
    with open(os.path.join(raw_dir, "evidence_features_day6.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            features_map[d["question_id"]] = [
                d["top1_score"], d["mean_score"], d["score_std"],
                d["margin_top1_top2"], d["score_range"],
                d["max_similarity"], d["mean_redundancy"]
            ]

    # 5. Train and calibrate controller strictly on Train split
    print("\nTraining Controller on 700 Train Queries...")
    X_train = np.array([features_map[qid] for qid in train_ids], dtype=np.float32)
    y_train = np.array([oracle_map[qid] for qid in train_ids], dtype=np.int32)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    clf.fit(X_train_scaled, y_train)

    train_acc = (clf.predict(X_train_scaled) == y_train).mean() * 100
    print(f"✔ Controller Train Accuracy: {train_acc:.2f}%")

    # Tune tau on Validation split
    X_val = np.array([features_map[qid] for qid in val_ids], dtype=np.float32)
    y_val = np.array([oracle_map[qid] for qid in val_ids], dtype=np.int32)
    X_val_scaled = scaler.transform(X_val)
    val_probs = clf.predict_proba(X_val_scaled)[:, 1]

    best_tau = 0.5
    best_val_f1 = -1.0
    for cand_tau in np.linspace(0.1, 0.9, 9):
        val_preds = (val_probs >= cand_tau).astype(int)
        # Check simulated answer F1 on val
        sim_f1 = np.mean([b3_map[qid]["f1"] if p == 1 else b2_map[qid]["f1"] for qid, p in zip(val_ids, val_preds)])
        if sim_f1 > best_val_f1:
            best_val_f1 = sim_f1
            best_tau = float(round(cand_tau, 2))

    print(f"✔ Tuned Optimal Probability Threshold τ on Validation: {best_tau} (Val F1: {best_val_f1:.4f})")

    # Save frozen controller package
    model_save_path = os.path.join(raw_dir, "controller_router.pkl")
    joblib.dump({"scaler": scaler, "model": clf, "tau": best_tau, "is_trained": True}, model_save_path)
    print(f"✔ Controller saved to {model_save_path}")

    # 6. Build Baseline B5 (Adaptive-RAG Question Complexity)
    print("\nAssembling Baseline B5 (Adaptive-RAG Complexity Routing)...")
    b5_results = []
    for qid in b2_map:
        q_item = b2_map[qid]
        q_text = q_item["question"]
        tokens = q_text.split()
        q_len = len(tokens)
        q_lower = q_text.lower()

        is_comparison = any(kw in q_lower for kw in ["both", "more", "less", "older", "younger", "between", "either", "same"])
        is_chain = any(kw in q_lower for kw in ["where was the", "who was the", "which film", "what team"]) and q_len > 14

        if is_comparison or is_chain or q_len > 18:
            chosen = dict(b3_map[qid])
            chosen["routed_tier"] = "Complex (k=10)"
        elif q_len <= 7 and not any(kw in q_lower for kw in ["who", "what", "where", "when"]):
            chosen = dict(b0_map[qid])
            chosen["routed_tier"] = "Simple (k=0)"
        else:
            chosen = dict(b2_map[qid])
            chosen["routed_tier"] = "Moderate (k=5)"

        chosen["baseline_id"] = "B5_Adaptive_RAG_Complexity"
        b5_results.append(chosen)

    with open(os.path.join(raw_dir, "b5_adaptive_rag_outputs.json"), "w") as f:
        json.dump(b5_results, f, indent=2)

    # 7. Build Proposed Method B6 (Evidence-State Controller)
    print("Assembling Proposed Method B6 (Evidence-State Adaptive-RAG)...")
    b6_results = []
    all_qids = list(b2_map.keys())
    X_all = np.array([features_map[qid] for qid in all_qids], dtype=np.float32)
    X_all_scaled = scaler.transform(X_all)
    all_probs = clf.predict_proba(X_all_scaled)[:, 1]

    for qid, p_more in zip(all_qids, all_probs):
        action = "MORE" if p_more >= best_tau else "STOP"
        if action == "MORE":
            chosen = dict(b3_map[qid])
        else:
            chosen = dict(b2_map[qid])

        chosen["baseline_id"] = "B6_Proposed_Adaptive_RAG"
        chosen["action_taken"] = action
        chosen["p_more"] = float(round(p_more, 4))
        chosen["evidence_features"] = features_map[qid]
        b6_results.append(chosen)

    with open(os.path.join(raw_dir, "b6_proposed_adaptive_rag_outputs.json"), "w") as f:
        json.dump(b6_results, f, indent=2)

    b5_map = {x["question_id"]: x for x in b5_results}
    b6_map = {x["question_id"]: x for x in b6_results}

    all_methods = [
        ("B0_Direct_LLM", b0_map),
        ("B1_Fixed_RAG_3", b1_map),
        ("B2_Fixed_RAG_5", b2_map),
        ("B3_Fixed_RAG_10", b3_map),
        ("B4_Simple_Score_Rule", b4_map),
        ("B5_Adaptive_RAG_Complexity", b5_map),
        ("B6_Proposed_Adaptive_RAG", b6_map),
    ]

    # Compute summary tables for: (A) Full 1,000 queries, (B) Frozen Test Split (150 queries)
    def compute_summary(q_subset, subset_name):
        rows = []
        for m_id, m_dict in all_methods:
            records = [m_dict[qid] for qid in q_subset]
            em = float(np.mean([1.0 if r["exact_match"] else 0.0 for r in records]))
            f1 = float(np.mean([r["f1"] for r in records]))
            avg_docs = float(np.mean([r["retrieved_count"] for r in records]))
            avg_lat = float(np.mean([r["total_latency"] for r in records]))
            avg_tok = float(np.mean([r["input_tokens"] + r["output_tokens"] for r in records]))
            avg_rec = float(np.mean([r.get("recall_at_retrieved", 0.0) for r in records]))
            avg_mrr = float(np.mean([r.get("mrr", 0.0) for r in records]))
            avg_sf = float(np.mean([r.get("supporting_fact_cov", 0.0) for r in records]))

            rows.append({
                "baseline_id": m_id,
                "exact_match": round(em, 4),
                "f1": round(f1, 4),
                "avg_retrieved_docs": round(avg_docs, 2),
                "retrieval_recall": round(avg_rec, 4),
                "mrr": round(avg_mrr, 4),
                "sf_coverage": round(avg_sf, 4),
                "avg_latency_sec": round(avg_lat, 4),
                "avg_total_tokens": round(avg_tok, 1),
                "total_queries": len(records)
            })
        return pd.DataFrame(rows)

    df_full = compute_summary(all_qids, "Full 1,000 Dataset")
    df_test = compute_summary(test_ids, "Frozen Test Split (N=150)")

    df_full.to_csv(os.path.join(tables_dir, "baselines_summary.csv"), index=False)
    df_test.to_csv(os.path.join(tables_dir, "test_split_summary.csv"), index=False)

    print("\n" + "="*80)
    print(f"{'METHOD':<28} | {'EM':<8} | {'F1':<8} | {'DOCS':<6} | {'TOKENS':<8} | {'LATENCY':<8}")
    print("="*80)
    print("--- [FROZEN TEST SPLIT: 150 UNSEEN QUERIES] ---")
    for _, row in df_test.iterrows():
        print(f"{row['baseline_id']:<28} | {row['exact_match']*100:>6.2f}% | {row['f1']:>8.4f} | {row['avg_retrieved_docs']:>6.2f} | {row['avg_total_tokens']:>8.1f} | {row['avg_latency_sec']:>7.2f}s")

    print("\n" + "="*80)
    print("--- [FULL DATASET: 1,000 QUERIES] ---")
    for _, row in df_full.iterrows():
        print(f"{row['baseline_id']:<28} | {row['exact_match']*100:>6.2f}% | {row['f1']:>8.4f} | {row['avg_retrieved_docs']:>6.2f} | {row['avg_total_tokens']:>8.1f} | {row['avg_latency_sec']:>7.2f}s")
    print("="*80)
    print(f"\nSaved updated results to:")
    print(f"  • {tables_dir}/baselines_summary.csv")
    print(f"  • {tables_dir}/test_split_summary.csv")


if __name__ == "__main__":
    main()
