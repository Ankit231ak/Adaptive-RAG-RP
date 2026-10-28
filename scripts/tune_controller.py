"""
Hyperparameter Tuning for GBDT Controller.
Uses strict train/validation splits, feature normalization, balanced F1 scoring,
and saves a fully compatible dictionary checkpoint.
"""
import os
import sys
import json
import logging
from typing import List, Dict, Any
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.common import load_config, set_seed
from scripts.train_controller import load_evidence_features, load_oracle_labels

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    print("=========================================================================")
    print("      HYPERPARAMETER TUNING: EVIDENCE CONTROLLER (GRID SEARCH)           ")
    print("=========================================================================\n")

    config = load_config("configs/base.yaml")
    set_seed(config.get("seed", 42))

    splits_dir = config["paths"]["splits_dir"]
    raw_dir = config["paths"]["results_raw_dir"]
    tables_dir = config["paths"]["results_tables_dir"]

    # 1. Load splits
    with open(os.path.join(splits_dir, "train_ids.json"), "r") as f:
        train_ids = json.load(f)
    with open(os.path.join(splits_dir, "val_ids.json"), "r") as f:
        val_ids = json.load(f)

    # 2. Load Features & Oracle
    features_map = load_evidence_features(tables_dir, raw_dir)
    oracle_map = load_oracle_labels(raw_dir, train_ids, val_ids)

    X_train = np.array([features_map[qid] for qid in train_ids if qid in features_map and qid in oracle_map], dtype=np.float32)
    y_train = np.array([1 if oracle_map[qid] == "MORE" else 0 for qid in train_ids if qid in features_map and qid in oracle_map], dtype=np.int32)

    X_val = np.array([features_map[qid] for qid in val_ids if qid in features_map and qid in oracle_map], dtype=np.float32)
    y_val = np.array([1 if oracle_map[qid] == "MORE" else 0 for qid in val_ids if qid in features_map and qid in oracle_map], dtype=np.int32)

    # 3. Fit Scaler strictly on train data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 4. Grid Search with Balanced Scoring
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [2, 3, 4],
        'min_samples_split': [2, 5, 10]
    }

    print(f"Running GridSearchCV over {len(X_train)} training samples (CV=5)...")
    base_model = GradientBoostingClassifier(random_state=42)
    
    # Use f1 or roc_auc to avoid majority class collapse on 87:13 imbalance
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    print(f"✔ Best CV ROC-AUC Score: {grid_search.best_score_:.4f}")
    print(f"✔ Optimal Parameters:    {grid_search.best_params_}")

    # Evaluate on unseen Validation Set
    val_probs = best_model.predict_proba(X_val_scaled)[:, 1]
    
    # Calibrate optimal threshold tau on validation
    best_tau = 0.3
    best_val_score = -1.0
    for tau_cand in np.linspace(0.1, 0.5, 9):
        preds = (val_probs >= tau_cand).astype(int)
        from sklearn.metrics import f1_score
        f1_m = f1_score(y_val, preds, zero_division=0)
        if f1_m > best_val_score:
            best_val_score = f1_m
            best_tau = round(float(tau_cand), 2)

    val_preds_final = (val_probs >= best_tau).astype(int)
    val_acc = (val_preds_final == y_val).mean() * 100
    stop_rate = (val_preds_final == 0).mean() * 100

    print(f"✔ Tuned Optimal Threshold Tau: {best_tau}")
    print(f"✔ Validation Accuracy:        {val_acc:.2f}%")
    print(f"✔ Validation Early STOP Rate: {stop_rate:.1f}%\n")

    # 5. Save in verified dictionary checkpoint format
    save_path = os.path.join(raw_dir, "controller_router.pkl")
    checkpoint = {
        "scaler": scaler,
        "model": best_model,
        "tau": best_tau,
        "backend": "gbdt",
        "is_trained": True,
        "best_params": grid_search.best_params_
    }
    joblib.dump(checkpoint, save_path)
    print(f"✔ Saved fully compatible checkpoint to: {save_path}")


if __name__ == "__main__":
    main()
