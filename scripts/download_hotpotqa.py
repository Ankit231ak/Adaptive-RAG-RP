import json
import os
from datasets import load_dataset

def convert_example(ex):
    return {
        "_id": ex["id"],
        "question": ex["question"],
        "answer": ex["answer"],
        "type": ex["type"],
        "level": ex["level"],
        "supporting_facts": [
            [t, s] for t, s in zip(ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"])
        ],
        "context": [
            [t, s_list] for t, s_list in zip(ex["context"]["title"], ex["context"]["sentences"])
        ],
    }

def main():
    target_dir = os.path.join("data", "hotpotqa")
    os.makedirs(target_dir, exist_ok=True)
    
    print("Loading HotpotQA dataset ('hotpotqa/hotpot_qa', 'distractor') from Hugging Face...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
    
    train_path = os.path.join(target_dir, "hotpot_train_v1.1.json")
    dev_path = os.path.join(target_dir, "hotpot_dev_distractor_v1.json")
    
    print(f"Processing train split ({len(ds['train'])} examples)...")
    train_data = [convert_example(ex) for ex in ds["train"]]
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
    print(f"Saved train set to {train_path} ({os.path.getsize(train_path) / (1024*1024):.2f} MB)")
    
    print(f"Processing dev split ({len(ds['validation'])} examples)...")
    dev_data = [convert_example(ex) for ex in ds["validation"]]
    with open(dev_path, "w", encoding="utf-8") as f:
        json.dump(dev_data, f, indent=2)
    print(f"Saved dev set to {dev_path} ({os.path.getsize(dev_path) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    main()
