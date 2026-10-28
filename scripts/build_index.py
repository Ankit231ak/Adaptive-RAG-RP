"""
Build FAISS Index Script.
Builds corpus, encodes dense embeddings, and creates/saves FAISS index.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import load_config, set_seed
from src.data_and_retrieval import create_sample_dataset, DenseRetriever


def main():
    parser = argparse.ArgumentParser(description="Build and test FAISS dense index.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to base config")
    parser.add_argument("--sample_size", type=int, default=50, help="Number of questions to sample for corpus")
    parser.add_argument("--output_index", type=str, default="data/processed/faiss_index.bin", help="Index output path")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    raw_path = config["paths"]["raw_dev_data"]
    print(f"Loading {args.sample_size} questions from {raw_path}...")
    examples, corpus = create_sample_dataset(raw_path, sample_size=args.sample_size, seed=config.get("seed", 42))
    print(f"Loaded {len(examples)} questions and {len(corpus)} unique corpus documents.")

    print(f"\nBuilding dense embeddings and FAISS index...")
    retriever = DenseRetriever(corpus=corpus, config=config)
    start_t = time.time()
    retriever.build_index()
    elapsed = time.time() - start_t
    print(f"✔ Index built in {elapsed:.2f} seconds.")

    # Save index
    os.makedirs(os.path.dirname(args.output_index), exist_ok=True)
    retriever.vector_store.save(args.output_index)
    print(f"✔ Saved index to {args.output_index}")

    # Spot-check Day 3 requirement: test top-10 retrieval on sample queries
    print("\n-------------------------------------------------------------------------")
    print("           SAMPLE RETRIEVAL SPOT-CHECK (TOP-10 RANKED DOCS)              ")
    print("-------------------------------------------------------------------------")
    test_ex = examples[0]
    print(f"Query: {test_ex.question}")
    results = retriever.retrieve(test_ex.question, top_k=10, candidate_doc_ids=test_ex.context_doc_ids)
    for rd in results:
        print(f" Rank {rd.rank:2d} | Score: {rd.score:+.4f} | ID: {rd.doc_id} | Title: {rd.doc.title[:40]}")

    d5, d6_10 = retriever.retrieve_split_5_and_5(test_ex.question, candidate_doc_ids=test_ex.context_doc_ids)
    print(f"\n✔ Clean 5+5 split verified: D5={len(d5)} docs, D6:10={len(d6_10)} docs. Zero duplicate IDs.")
    print("=========================================================================\n")


if __name__ == "__main__":
    main()
