import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.common import load_config
from src.method_and_controller import create_oracle_record

def build_oracle():
    print("Building Oracle Labels...")
    config = load_config()
    raw_dir = config["paths"]["results_raw_dir"]
    
    # Load B2 (k=5) and B3 (k=10) outputs
    b2_path = os.path.join(raw_dir, "b2_fixed_rag_5_outputs.json")
    b3_path = os.path.join(raw_dir, "b3_fixed_rag_10_outputs.json")
    
    with open(b2_path, "r") as f:
        b2_data = {item["question_id"]: item for item in json.load(f)}
    
    with open(b3_path, "r") as f:
        b3_data = {item["question_id"]: item for item in json.load(f)}
        
    delta_threshold = config["controller"]["delta_threshold"]
    
    oracle_records = []
    stop_count = 0
    more_count = 0
    
    for q_id, b2_item in b2_data.items():
        if q_id not in b3_data:
            continue
            
        b3_item = b3_data[q_id]
        
        record = create_oracle_record(
            question_id=q_id,
            question=b2_item["question"],
            gold_answer=b2_item["gold_answer"],
            ans_d5=b2_item["predicted_answer"],
            ans_d10=b3_item["predicted_answer"],
            f1_d5=b2_item["f1"],
            f1_d10=b3_item["f1"],
            delta_threshold=delta_threshold
        )
        
        if record["oracle_label"] == "STOP":
            stop_count += 1
        else:
            more_count += 1
            
        oracle_records.append(record)
        
    out_path = os.path.join(raw_dir, "oracle_labels.json")
    with open(out_path, "w") as f:
        json.dump(oracle_records, f, indent=2)
        
    print(f"Generated {len(oracle_records)} oracle labels.")
    print(f"  STOP labels: {stop_count}")
    print(f"  MORE labels: {more_count}")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    build_oracle()
