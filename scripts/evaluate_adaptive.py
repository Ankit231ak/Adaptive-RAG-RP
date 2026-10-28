"""
Adaptive RAG Evaluation Script.
Evaluates the adaptive stopping pipeline against test questions, logging F1,
token consumption, latency, and controller STOP/MORE decisions.
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.common import load_config, set_seed
from src.data_and_retrieval import create_sample_dataset, DenseRetriever
from src.generation_and_eval import GeneratorWrapper
from src.method_and_controller import ControllerRouter
from src.pipeline import AdaptiveRAGPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_evaluation(num_samples: int = 50, seed: int = 42):
    print("=========================================================================")
    print("                 ADAPTIVE RAG PIPELINE EVALUATION                        ")
    print("=========================================================================\n")

    config = load_config("configs/base.yaml")
    set_seed(seed)

    test_file = config["paths"]["raw_dev_data"]
    print(f"Loading {num_samples} sample questions from {test_file} (Seed: {seed})...")
    questions, corpus = create_sample_dataset(
        raw_file_path=test_file,
        sample_size=num_samples,
        seed=seed
    )

    print("Building dense retriever and indexing passages...")
    retriever = DenseRetriever(corpus=corpus, config=config)
    retriever.build_index()

    generator = GeneratorWrapper(config=config)

    # Load trained controller
    router = ControllerRouter(tau=config.get("controller", {}).get("tau", 0.3))
    router_path = os.path.join(config["paths"]["results_raw_dir"], "controller_router.pkl")
    if os.path.exists(router_path):
        router.load(router_path)
        logger.info(f"Loaded trained controller checkpoint from {router_path}")
    else:
        logger.warning(f"Checkpoint {router_path} not found. Running with default controller.")

    pipeline = AdaptiveRAGPipeline(
        corpus=corpus,
        retriever=retriever,
        generator=generator,
        router=router,
        config=config
    )

    print(f"\nRunning evaluation over {len(questions)} queries...")
    results = []
    for idx, q in enumerate(questions, 1):
        res = pipeline.run_single(q)
        results.append(res)
        q_snip = q.question[:45] + "..." if len(q.question) > 45 else q.question
        print(f"  [{idx:02d}/{len(questions):02d}] Action: {res.action_taken:<4} (P={res.p_more:.3f}) | F1: {res.f1:.2f} | EM: {int(res.exact_match)} | {q_snip}")

    # Aggregated Summary
    avg_f1 = sum(r.f1 for r in results) / len(results) if results else 0.0
    avg_em = (sum(1 for r in results if r.exact_match) / len(results)) * 100 if results else 0.0
    total_tokens = sum(r.input_tokens + r.output_tokens for r in results)
    avg_tokens = total_tokens / len(results) if results else 0.0
    stop_count = sum(1 for r in results if r.action_taken == "STOP")
    more_count = sum(1 for r in results if r.action_taken == "MORE")
    stop_rate = (stop_count / len(results)) * 100 if results else 0.0

    print("\n-------------------------------------------------------------------------")
    print("                    EVALUATION METRICS SUMMARY                           ")
    print("-------------------------------------------------------------------------")
    print(f"• Total Evaluated Queries : {len(results)}")
    print(f"• Exact Match (EM)        : {avg_em:.2f}%")
    print(f"• Average F1 Score        : {avg_f1:.4f}")
    print(f"• Early STOP Rate         : {stop_rate:.1f}% ({stop_count} STOPs, {more_count} MOREs)")
    print(f"• Average Tokens / Query  : {avg_tokens:.1f}")
    print("-------------------------------------------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Adaptive RAG Pipeline")
    parser.add_argument("--samples", "-n", type=int, default=30, help="Number of questions to evaluate")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    run_evaluation(num_samples=args.samples, seed=args.seed)
