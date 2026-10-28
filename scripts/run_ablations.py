"""
Essential Ablations (A1 to A6) and Oracle Upper Bound (E5) Runner.
Evaluates the marginal contribution of each evidence-state feature, controller architecture,
delta threshold, and tau sensitivity on the frozen test set.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.common.cache_builder import ensure_baseline_caches


def main():
    print("=========================================================================")
    print("          ADAPTIVE-RAG: ESSENTIAL ABLATIONS & EXPERIMENT MATRIX          ")
    print("=========================================================================\n")

    raw_dir = "results/raw"
    tables_dir = "results/tables"

    # Ensure raw caches exist, reconstructing them if missing
    ensure_baseline_caches(raw_dir=raw_dir, tables_dir=tables_dir, splits_dir="data/splits")

    # Load splits
    with open("data/splits/train_ids.json") as f:
        train_ids = json.load(f)
    with open("data/splits/test_ids.json") as f:
        test_ids = json.load(f)

    # Load baseline outputs
    with open(os.path.join(raw_dir, "b2_fixed_rag_5_outputs.json")) as f:
        b2_map = {x["question_id"]: x for x in json.load(f)}
    with open(os.path.join(raw_dir, "b3_fixed_rag_10_outputs.json")) as f:
        b3_map = {x["question_id"]: x for x in json.load(f)}

    # Feature definitions
    FEATURE_NAMES = [
        "top1_score", "mean_score", "score_std",
        "margin_top1_top2", "score_range",
        "max_similarity", "mean_redundancy"
    ]

    features_map = {}
    with open(os.path.join(raw_dir, "evidence_features_day6.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            features_map[d["question_id"]] = [d[k] for k in FEATURE_NAMES]

    ablation_results = []

    def evaluate_controller(feat_indices, model_type="gbdt", delta_val=0.05, tau_val=0.3, label_desc=""):
        # Build Oracle for current delta
        oracle_train = {}
        for qid in train_ids:
            delta_f1 = b3_map[qid]["f1"] - b2_map[qid]["f1"]
            oracle_train[qid] = 1 if delta_f1 > delta_val else 0

        X_tr = np.array([[features_map[qid][i] for i in feat_indices] for qid in train_ids], dtype=np.float32)
        y_tr = np.array([oracle_train[qid] for qid in train_ids], dtype=np.int32)

        X_te = np.array([[features_map[qid][i] for i in feat_indices] for qid in test_ids], dtype=np.float32)

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        if model_type == "gbdt":
            clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        elif model_type == "logistic_regression":
            clf = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
        elif model_type == "random_forest":
            clf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42)
        else:
            raise ValueError(f"Unknown model {model_type}")

        clf.fit(X_tr_sc, y_tr)
        probs = clf.predict_proba(X_te_sc)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X_te_sc)
        preds = (probs >= tau_val).astype(int)

        # Simulate B6 on test
        sim_f1 = []
        sim_em = []
        sim_tok = []
        sim_docs = []
        stops = 0
        mores = 0

        for qid, p in zip(test_ids, preds):
            if p == 1:
                rec = b3_map[qid]
                mores += 1
            else:
                rec = b2_map[qid]
                stops += 1
            sim_f1.append(rec["f1"])
            sim_em.append(1.0 if rec["exact_match"] else 0.0)
            sim_tok.append(rec["input_tokens"] + rec["output_tokens"])
            sim_docs.append(rec["retrieved_count"])

        res = {
            "ablation_id": label_desc,
            "f1": round(float(np.mean(sim_f1)), 4),
            "exact_match": round(float(np.mean(sim_em)), 4),
            "avg_tokens": round(float(np.mean(sim_tok)), 1),
            "avg_docs": round(float(np.mean(sim_docs)), 2),
            "stop_count": stops,
            "more_count": mores,
            "stop_rate": round(stops / len(test_ids), 3)
        }
        return res

    all_idx = list(range(7))

    # A0: Full Proposed Model
    ablation_results.append(evaluate_controller(all_idx, model_type="gbdt", label_desc="A0: Full V1 Features (Proposed GBDT)"))

    # A1: Remove Redundancy (features without mean_redundancy)
    no_red_idx = [0, 1, 2, 3, 4, 5]
    ablation_results.append(evaluate_controller(no_red_idx, model_type="gbdt", label_desc="A1: Drop Redundancy (No mean_redundancy)"))

    # A2: Remove Score / Margin features (keep only similarity & redundancy)
    sim_red_only = [5, 6]
    ablation_results.append(evaluate_controller(sim_red_only, model_type="gbdt", label_desc="A2: Drop Score/Margin (Only sim & redundancy)"))

    # A3: Score features only (drop redundancy & max_sim)
    score_only = [0, 1, 2, 3, 4]
    ablation_results.append(evaluate_controller(score_only, model_type="gbdt", label_desc="A3: Score Features Only (Drop sim & redundancy)"))

    # A4: Model architecture comparison
    ablation_results.append(evaluate_controller(all_idx, model_type="logistic_regression", label_desc="A4.1: Controller = Logistic Regression"))
    ablation_results.append(evaluate_controller(all_idx, model_type="random_forest", label_desc="A4.2: Controller = Random Forest"))

    # A5: Delta sensitivity sweep (delta = 0.00, 0.02, 0.05, 0.10)
    for d_val in [0.00, 0.02, 0.10]:
        ablation_results.append(evaluate_controller(all_idx, model_type="gbdt", delta_val=d_val, label_desc=f"A5: Delta Sensitivity (delta={d_val})"))

    # A6: Tau sensitivity sweep (tau = 0.1, 0.3, 0.5, 0.7, 0.9)
    for t_val in [0.1, 0.2, 0.4, 0.5, 0.7, 0.9]:
        ablation_results.append(evaluate_controller(all_idx, model_type="gbdt", tau_val=t_val, label_desc=f"A6: Tau Sensitivity (tau={t_val})"))

    # E5: Oracle Upper Bound (Perfect foresight: choose k=10 iff delta_f1 > 0)
    oracle_f1 = []
    oracle_em = []
    oracle_tok = []
    oracle_docs = []
    o_stops = 0
    o_mores = 0
    for qid in test_ids:
        delta_f1 = b3_map[qid]["f1"] - b2_map[qid]["f1"]
        if delta_f1 > 0:
            rec = b3_map[qid]
            o_mores += 1
        else:
            rec = b2_map[qid]
            o_stops += 1
        oracle_f1.append(rec["f1"])
        oracle_em.append(1.0 if rec["exact_match"] else 0.0)
        oracle_tok.append(rec["input_tokens"] + rec["output_tokens"])
        oracle_docs.append(rec["retrieved_count"])

    ablation_results.append({
        "ablation_id": "E5: Oracle Upper Bound (Perfect Hindsight)",
        "f1": round(float(np.mean(oracle_f1)), 4),
        "exact_match": round(float(np.mean(oracle_em)), 4),
        "avg_tokens": round(float(np.mean(oracle_tok)), 1),
        "avg_docs": round(float(np.mean(oracle_docs)), 2),
        "stop_count": o_stops,
        "more_count": o_mores,
        "stop_rate": round(o_stops / len(test_ids), 3)
    })

    df_abl = pd.DataFrame(ablation_results)
    out_file = os.path.join(tables_dir, "ablations_summary.csv")
    df_abl.to_csv(out_file, index=False)

    print("\n" + "="*85)
    print(f"{'ABLATION / EXPERIMENT':<42} | {'F1':<8} | {'EM':<8} | {'TOKENS':<8} | {'STOP%':<6}")
    print("="*85)
    for _, row in df_abl.iterrows():
        print(f"{row['ablation_id']:<42} | {row['f1']:>8.4f} | {row['exact_match']*100:>6.2f}% | {row['avg_tokens']:>8.1f} | {row['stop_rate']*100:>5.1f}%")
    print("="*85)
    print(f"\nSaved ablations summary to {out_file}")


if __name__ == "__main__":
    main()
