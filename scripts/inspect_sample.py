"""
Script to load a tiny HotpotQA sample, inspect all required fields,
and verify corpus representation with stable document IDs.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import load_config, set_seed
from src.data_and_retrieval import create_sample_dataset, generate_stable_doc_id


def inspect_sample():
    config = load_config("configs/base.yaml")
    set_seed(config.get("seed", 42))

    raw_dev_path = config["paths"]["raw_dev_data"]
    tiny_size = config["sample"]["tiny_sample_size"]

    print("=========================================================")
    print(f"Loading tiny HotpotQA sample ({tiny_size} examples)...")
    print(f"Source file: {raw_dev_path}")
    print("=========================================================\n")

    examples, corpus = create_sample_dataset(
        raw_file_path=raw_dev_path,
        sample_size=tiny_size,
        seed=config.get("seed", 42)
    )

    print(f"Successfully loaded {len(examples)} examples.")
    print(f"Total unique documents in corpus: {len(corpus)}\n")

    ex0 = examples[0]
    print("---------------------------------------------------------")
    print(" FIELD INSPECTION (Sample #1)")
    print("---------------------------------------------------------")
    print(f"• ID (_id)             : {ex0.id}")
    print(f"• Question             : {ex0.question}")
    print(f"• Answer               : {ex0.answer}")
    print(f"• Type                 : {ex0.type}")
    print(f"• Level                : {ex0.level}")
    print(f"• Context Document Count: {len(ex0.context_doc_ids)}")
    print(f"• Gold Document Count   : {len(ex0.gold_doc_ids)}\n")

    print("--- SUPPORTING FACTS ---")
    for i, sf in enumerate(ex0.supporting_facts):
        print(f"  [{i+1}] Title: '{sf.title}' | Sentence Index: {sf.sent_idx} | Stable Doc ID: {sf.doc_id}")

    print("\n--- CONTEXT DOCUMENTS & STABLE IDs (Top 3) ---")
    for i, doc_id in enumerate(ex0.context_doc_ids[:3]):
        doc = corpus.get_document(doc_id)
        print(f"\n  Document [{i+1}]")
        print(f"  • Stable ID : {doc.doc_id}")
        print(f"  • Title     : {doc.title}")
        print(f"  • Sentences : {len(doc.sentences)} sentences")
        print(f"  • Text Snippet: {doc.text[:120]}...")

    print("\n=========================================================")
    print(" STABLE ID DETERMINISM VERIFICATION")
    print("=========================================================")
    doc0 = corpus.get_document(ex0.context_doc_ids[0])
    recomputed_id = generate_stable_doc_id(doc0.title, doc0.text)
    print(f"Original Stable ID : {doc0.doc_id}")
    print(f"Recomputed ID      : {recomputed_id}")
    assert doc0.doc_id == recomputed_id, "ID Mismatch! Hash function is not deterministic."
    print("✔ VERIFICATION PASSED: Stable Document IDs are 100% deterministic.\n")

    splits_dir = config["paths"]["splits_dir"]
    processed_dir = config["paths"]["processed_dir"]
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    splits_out_path = os.path.join(splits_dir, "tiny_sample_dev.json")
    corpus_out_path = os.path.join(processed_dir, "tiny_corpus.json")

    with open(splits_out_path, "w", encoding="utf-8") as f:
        json.dump([ex.to_dict() for ex in examples], f, indent=2)

    with open(corpus_out_path, "w", encoding="utf-8") as f:
        json.dump(corpus.to_list(), f, indent=2)

    print(f"Saved tiny sample dataset split to : {splits_out_path}")
    print(f"Saved extracted tiny corpus to      : {corpus_out_path}")
    print("=========================================================")


if __name__ == "__main__":
    inspect_sample()
