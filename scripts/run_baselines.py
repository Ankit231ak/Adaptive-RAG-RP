"""
Run Baselines Runner (Days 4 & 5 Milestone).
Runs Direct LLM (B0), Fixed RAG-3 (B1), Fixed RAG-5 (B2), Fixed RAG-10 (B3), Simple Score Rule (B4).
Outputs per-query JSON logs and summary comparison CSV.
"""
import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import load_config, set_seed
from src.data_and_retrieval import create_sample_dataset, DenseRetriever
from src import GeneratorWrapper, evaluate_predictions
from src.pipeline.baselines import BaselineRunner, BaselineResult


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluation suite.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--sample_size", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("seed", 42)
    set_seed(seed)

    print("=========================================================================")
    print("             ADAPTIVE RAG — BASELINE SUITE EXPERIMENTS                   ")
    print("=========================================================================\n")

    raw_path = config["paths"]["raw_dev_data"]
    print(f"Loading {args.sample_size} evaluation examples from {raw_path}...")
    examples, corpus = create_sample_dataset(raw_path, sample_size=args.sample_size, seed=seed)
    print(f"Loaded {len(examples)} questions and {len(corpus)} corpus documents.")

    print("\nInitializing FAISS dense retriever and generator wrapper...")
    retriever = DenseRetriever(corpus=corpus, config=config)
    retriever.build_index()

    generator = GeneratorWrapper(config=config)
    runner = BaselineRunner(retriever=retriever, generator=generator, config=config)

    baselines_to_run = [
        ("B0_Direct_LLM", lambda ex: runner.run_direct_llm(ex)),
        ("B1_Fixed_RAG_3", lambda ex: runner.run_fixed_rag(ex, k=3)),
        ("B2_Fixed_RAG_5", lambda ex: runner.run_fixed_rag(ex, k=5)),
        ("B3_Fixed_RAG_10", lambda ex: runner.run_fixed_rag(ex, k=10)),
        ("B4_Simple_Score_Rule", lambda ex: runner.run_simple_score_rule(ex, threshold=0.65)),
    ]

    all_baseline_results: Dict[str, List[BaselineResult]] = {}
    summary_rows = []

    os.makedirs(os.path.join(args.output_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "tables"), exist_ok=True)

    for b_id, run_fn in baselines_to_run:
        print(f"\nRunning Baseline: {b_id} on {len(examples)} queries...")
        b_results: List[BaselineResult] = []
        preds = []
        gts = []

        for ex in examples:
            res = run_fn(ex)
            b_results.append(res)
            preds.append(res.predicted_answer)
            gts.append(res.gold_answer)

        all_baseline_results[b_id] = b_results
        eval_metrics = evaluate_predictions(preds, gts)

        avg_latency = float(sum(r.total_latency for r in b_results) / len(b_results))
        avg_tokens = float(sum(r.input_tokens + r.output_tokens for r in b_results) / len(b_results))
        avg_retrieved = float(sum(r.retrieved_count for r in b_results) / len(b_results))
        avg_recall = float(sum(r.recall_at_retrieved for r in b_results) / len(b_results))
        avg_mrr = float(sum(r.mrr for r in b_results) / len(b_results))
        avg_sf_cov = float(sum(r.supporting_fact_cov for r in b_results) / len(b_results))

        summary_rows.append({
            "baseline_id": b_id,
            "exact_match": eval_metrics["exact_match"],
            "f1": eval_metrics["f1"],
            "avg_retrieved_docs": round(avg_retrieved, 2),
            "retrieval_recall": round(avg_recall, 4),
            "mrr": round(avg_mrr, 4),
            "sf_coverage": round(avg_sf_cov, 4),
            "avg_latency_sec": round(avg_latency, 4),
            "avg_total_tokens": round(avg_tokens, 1),
            "total_queries": len(examples)
        })

        # Save raw per-baseline log
        raw_baseline_file = os.path.join(args.output_dir, "raw", f"{b_id.lower()}_outputs.json")
        with open(raw_baseline_file, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in b_results], f, indent=2)

    # Save summary CSV
    csv_file = os.path.join(args.output_dir, "tables", "baselines_summary.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "baseline_id", "exact_match", "f1", "avg_retrieved_docs",
            "retrieval_recall", "mrr", "sf_coverage", "avg_latency_sec", "avg_total_tokens", "total_queries"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n-------------------------------------------------------------------------")
    print("                    BASELINES COMPARISON SUMMARY TABLE                   ")
    print("-------------------------------------------------------------------------")
    print(f"{'Baseline ID':<22} | {'EM (%)':<8} | {'F1 (%)':<8} | {'Avg Docs':<9} | {'Recall':<8} | {'Latency':<9}")
    print("-" * 73)
    for row in summary_rows:
        print(
            f"{row['baseline_id']:<22} | "
            f"{row['exact_match'] * 100:6.2f}% | "
            f"{row['f1'] * 100:6.2f}% | "
            f"{row['avg_retrieved_docs']:8.2f}  | "
            f"{row['retrieval_recall']:7.4f}  | "
            f"{row['avg_latency_sec']:7.4f}s"
        )
    print("-------------------------------------------------------------------------")
    print(f"✔ Summary saved to: {csv_file}")
    print("=========================================================================\n")


if __name__ == "__main__":
    main()
