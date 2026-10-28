import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.common import load_config
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def run_model_selection():
    print("Running Model Selection Bake-off with Train/Test Split (80/20)...")
    config = load_config()
    raw_dir = config["paths"]["results_raw_dir"]
    
    oracle_path = os.path.join(raw_dir, "oracle_labels.json")
    with open(oracle_path, "r") as f:
        oracle_data = {item["question_id"]: item["oracle_label"] for item in json.load(f)}
        
    features_path = os.path.join(raw_dir, "evidence_features_day6.jsonl")
    
    X = []
    y_str = []
    
    with open(features_path, "r") as f:
        for line in f:
            data = json.loads(line)
            q_id = data["question_id"]
            if q_id in oracle_data:
                features = [
                    data["top1_score"], data["mean_score"], data["score_std"],
                    data["margin_top1_top2"], data["score_range"], 
                    data["max_similarity"], data["mean_redundancy"]
                ]
                X.append(features)
                y_str.append(oracle_data[q_id])
                
    X = np.array(X, dtype=np.float32)
    y = np.array([1 if lbl == "MORE" else 0 for lbl in y_str], dtype=np.int32)
    
    # 80/20 Train-Test Split (800 for training, 200 for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features (Fit ONLY on training data to prevent data leakage!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=500, random_state=42),
        "Random Forest": RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42, max_depth=5),
        "Support Vector Machine": SVC(class_weight="balanced", probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=3)
    }
    
    print("\n" + "="*70)
    print(f"{'Machine Learning Model':<25} | {'Train Accuracy':<15} | {'Test Accuracy':<15}")
    print("="*70)
    
    for name, model in models.items():
        # Train on the 80% split
        model.fit(X_train_scaled, y_train)
        
        # Evaluate on Train
        train_preds = model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_preds) * 100
        
        # Evaluate on Test (Unseen data)
        test_preds = model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, test_preds) * 100
        
        print(f"{name:<25} | {train_acc:>13.2f}% | {test_acc:>13.2f}%")
    print("="*70)

if __name__ == "__main__":
    run_model_selection()
