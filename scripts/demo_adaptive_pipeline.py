"""
Adaptive RAG — Interactive Live Demonstration Script for Faculty / Presentations.
Demonstrates the 4-stage adaptive retrieval workflow:
  Stage 1: Initial Dense Retrieval (k=5)
  Stage 2: 7-Dimensional Evidence-State Feature Extraction (X_5 in R^7)
  Stage 3: GBDT Meta-Classifier Decision Gate (STOP vs MORE, tau=0.30)
  Stage 4: Context Assembly & Answer Generation (with token/latency audit)
"""

import argparse
import logging
import os
import sys
import time
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.common import load_config
from src.data_and_retrieval import create_sample_dataset, DenseRetriever
from src.generation_and_eval import GeneratorWrapper, exact_match_score, f1_score
from src.method_and_controller import ControllerRouter, extract_evidence_features

logging.basicConfig(level=logging.ERROR)

# ANSI Color Codes for Terminal Styling
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
WHITE = "\033[97m"
BLUE = "\033[94m"
GRAY = "\033[90m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_banner():
    print(CYAN + BOLD + "\n" + "=" * 76)
    print("   ADAPTIVE-RAG : POST-RETRIEVAL EVIDENCE-STATE CONTROLLER PIPELINE")
    print("        Live Interactive Demonstration for Faculty & Technical Review")
    print("=" * 76 + RESET + "\n")


def run_demo(selected_sample: Optional[int] = None, custom_query: Optional[str] = None, seed: Optional[int] = None):
    print_banner()

    config = load_config("configs/base.yaml")
    test_file = config["paths"]["raw_dev_data"]

    import random
    if seed is None:
        import random as rand_mod
        active_seed = rand_mod.randint(100, 99999)
    else:
        active_seed = seed

    print(YELLOW + f"▶ [INITIALIZING] Loading corpus sample and models (Seed: {active_seed})..." + RESET)
    # Load sample questions to demonstrate real multi-hop queries
    questions, corpus = create_sample_dataset(test_file, sample_size=50, seed=active_seed)

    # Initialize Dense Retriever with project config (SentenceTransformers MiniLM or Ollama)
    retriever = DenseRetriever(corpus=corpus, config=config)
    embedder = retriever.embedding_model
    retriever.build_index()

    # Generator with graceful fallback
    generator = GeneratorWrapper(config=config)

    # Load trained GBDT Controller
    router = ControllerRouter(tau=0.3)
    router_path = os.path.join(config["paths"]["results_raw_dir"], "controller_router.pkl")
    if os.path.exists(router_path):
        router.load(router_path)
    else:
        print(GRAY + "  (Using default calibrated decision gate, tau=0.30)" + RESET)

    print(GREEN + "✔ Pipeline initialized and FAISS vector index ready.\n" + RESET)

    # Curate 3 diverse questions (e.g. Bridge and Comparison)
    bridge_questions = [q for q in questions if getattr(q, "type", "") == "bridge"]
    comp_questions = [q for q in questions if getattr(q, "type", "") == "comparison"]
    sample_options = []
    if bridge_questions:
        sample_options.append(random.choice(bridge_questions))
    if comp_questions:
        sample_options.append(random.choice(comp_questions))
    for q in questions:
        if len(sample_options) >= 3:
            break
        if q not in sample_options:
            sample_options.append(q)

    gold_answer = None
    question_type = "Multi-Hop Reasoning"

    if custom_query:
        user_q = custom_query.strip()
    elif selected_sample is not None and 1 <= selected_sample <= len(sample_options):
        q_obj = sample_options[selected_sample - 1]
        user_q = q_obj.question
        gold_answer = q_obj.answer
        question_type = getattr(q_obj, "type", getattr(q_obj, "question_type", "Multi-Hop QA"))
    else:
        print(BOLD + "Select a question to run through the adaptive pipeline:" + RESET)
        for idx, q in enumerate(sample_options, 1):
            q_type = getattr(q, "type", getattr(q, "question_type", "Multi-hop"))
            print(f"  {CYAN}[{idx}]{RESET} {WHITE}{q.question}{RESET} {GRAY}({q_type}){RESET}")
        print(f"  {CYAN}[Enter]{RESET} {GRAY}Pick a random unseen question from the test corpus{RESET}")
        print(f"  {CYAN}[Or type]{RESET} {GRAY}Any custom natural language question{RESET}\n")

        raw_in = input(BOLD + WHITE + "Enter choice [1/2/3], custom question, or press Enter: " + RESET).strip()

        if raw_in in ("1", "2", "3") and int(raw_in) <= len(sample_options):
            choice_idx = int(raw_in) - 1
            q_obj = sample_options[choice_idx]
            user_q = q_obj.question
            gold_answer = q_obj.answer
            question_type = getattr(q_obj, "type", getattr(q_obj, "question_type", "Multi-Hop"))
        elif not raw_in:
            q_obj = random.choice(questions)
            user_q = q_obj.question
            gold_answer = q_obj.answer
            question_type = getattr(q_obj, "type", getattr(q_obj, "question_type", "Multi-Hop"))
        else:
            user_q = raw_in
            gold_answer = None
            question_type = "Custom User Question"

    print(CYAN + BOLD + f"\n[ACTIVE QUESTION] : {WHITE}{user_q}" + RESET)
    print(GRAY + f"Question Category : {question_type}" + RESET)
    if gold_answer:
        print(GRAY + f"Ground-Truth Gold : {gold_answer}" + RESET)

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 1: Initial Retrieval (k=5)
    # ──────────────────────────────────────────────────────────────────────────
    print(BLUE + BOLD + "\n" + "─" * 76)
    print("▶ STAGE 1: INITIAL DENSE RETRIEVAL (k = 5)")
    print("─" * 76 + RESET)
    t0 = time.time()

    query_emb = embedder.embed_query(user_q)
    top5_docs = retriever.retrieve(user_q, top_k=5)
    t_ret1 = time.time() - t0

    print(f"  Retrieved top-5 passages from FAISS index in {t_ret1:.3f}s:")
    for d in top5_docs:
        snippet = d.doc.text[:75].replace("\n", " ") + "..."
        print(f"    Rank {d.rank}: [{CYAN}Score: {d.score:.4f}{RESET}] {WHITE}{d.doc.title}{RESET} {GRAY}— {snippet}{RESET}")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 2: Evidence State Feature Extraction
    # ──────────────────────────────────────────────────────────────────────────
    print(BLUE + BOLD + "\n" + "─" * 76)
    print("▶ STAGE 2: POST-RETRIEVAL EVIDENCE-STATE FEATURE EXTRACTION (X_5 in R^7)")
    print("─" * 76 + RESET)
    time.sleep(0.4)

    scores = [rd.score for rd in top5_docs]
    doc_embs = np.array([rd.embedding for rd in top5_docs if rd.embedding is not None], dtype=np.float32)
    feat = extract_evidence_features(scores, doc_embs, query_emb)

    print("  7 Information-Dense Features Extracted from Vector Distribution:")
    print(f"    1. Top-1 Cosine Score   (f1): {CYAN}{feat.top1_score:.4f}{RESET}  {GRAY}(Primary confidence anchor){RESET}")
    print(f"    2. Mean Cosine Score    (f2): {CYAN}{feat.mean_score:.4f}{RESET}  {GRAY}(Average passage alignment){RESET}")
    print(f"    3. Score Variance / Std (f3): {CYAN}{feat.score_std:.4f}{RESET}  {GRAY}(Retrieved score spread){RESET}")
    print(f"    4. Margin (Top1 - Top2) (f4): {CYAN}{feat.margin_top1_top2:.4f}{RESET}  {GRAY}(First-to-second decay){RESET}")
    print(f"    5. Score Range (Top1-5) (f5): {CYAN}{feat.score_range:.4f}{RESET}  {GRAY}(Total drop across 5 ranks){RESET}")
    print(f"    6. Max Doc Similarity   (f6): {CYAN}{feat.max_similarity:.4f}{RESET}  {GRAY}(Max pairwise passage overlap){RESET}")
    print(f"    7. Mean Redundancy      (f7): {CYAN}{feat.mean_redundancy:.4f}{RESET}  {GRAY}(Mean pairwise overlap){RESET}")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 3: ML Controller Routing Decision
    # ──────────────────────────────────────────────────────────────────────────
    print(MAGENTA + BOLD + "\n" + "─" * 76)
    print("▶ STAGE 3: GBDT META-CLASSIFIER DECISION GATE (Decision Threshold tau = 0.30)")
    print("─" * 76 + RESET)
    time.sleep(0.4)

    features_list = [
        feat.top1_score,
        feat.mean_score,
        feat.score_std,
        feat.margin_top1_top2,
        feat.score_range,
        feat.max_similarity,
        feat.mean_redundancy,
    ]
    action, prob_more = router.predict_action(features_list)

    print(f"  GBDT Evaluation: P(MORE) = {prob_more:.4f}  |  Calibrated Threshold tau = 0.3000")

    if action == "STOP":
        print(GREEN + BOLD + f"  → DECISION: [STOP] (Confidence: {1.0 - prob_more:.1%})" + RESET)
        print(GREEN + "  → SCIENTIFIC RATIONALE: Evidence in top-5 is sufficiently aligned. Terminating retrieval early:" + RESET)
        print(GRAY + "      • Saves 5 extra passage fetches & protects against context bloat." + RESET)
        print(GRAY + "      • Shields LLM from distractor passages in ranks 6–10 (prevents -0.048 F1 collapse)." + RESET)
        print(GRAY + "      • Cuts token usage by ~48% and reduces inference latency by ~29%." + RESET)
        final_docs = top5_docs
    else:
        print(RED + BOLD + f"  → DECISION: [MORE] (Confidence: {prob_more:.1%})" + RESET)
        print(YELLOW + "  → SCIENTIFIC RATIONALE: High uncertainty or steep score decay in top-5. Triggering Stage 3.5 expansion:" + RESET)
        print(GRAY + "      • Expanding retrieval depth to k=10 to find missing multi-hop bridge documents." + RESET)
        t_exp = time.time()
        final_docs = retriever.retrieve(user_q, top_k=10)
        print(f"      • Retrieved additional D_6:10 in {time.time() - t_exp:.3f}s (Total: {len(final_docs)} documents).")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 4: Generator Output & Evaluation
    # ──────────────────────────────────────────────────────────────────────────
    print(BLUE + BOLD + "\n" + "─" * 76)
    print(f"▶ STAGE 4: ANSWER GENERATION (Feeding {len(final_docs)} Curated Passages to LLM)")
    print("─" * 76 + RESET)

    gen_start = time.time()
    docs_only = [rd.doc for rd in final_docs]
    gen_out = generator.generate(user_q, docs_only)
    gen_time = time.time() - gen_start

    print(WHITE + BOLD + f"\n  [PREDICTED ANSWER] : {gen_out.answer}" + RESET)
    if gold_answer:
        em = exact_match_score(gen_out.answer, gold_answer)
        f1, _, _ = f1_score(gen_out.answer, gold_answer)
        em_color = GREEN if em else RED
        print(f"  [GOLD ANSWER]      : {gold_answer}")
        print(f"  [SCORE METRICS]    : Exact Match: {em_color}{em}{RESET} | Token F1: {GREEN}{f1:.4f}{RESET}")

    total_tokens = gen_out.input_tokens + gen_out.output_tokens
    total_time = time.time() - t0

    # ──────────────────────────────────────────────────────────────────────────
    # SUMMARY / COMPARISON FOR FACULTY
    # ──────────────────────────────────────────────────────────────────────────
    print(CYAN + BOLD + "\n" + "=" * 76)
    print("                    EFFICIENCY & COST AUDIT SUMMARY")
    print("=" * 76 + RESET)
    fixed10_tokens_est = 1544
    tokens_saved_pct = max(0.0, (fixed10_tokens_est - total_tokens) / fixed10_tokens_est * 100) if action == "STOP" else 0.0

    print(f"  • Retrieval Decision : {BOLD}{action}{RESET} (Passages Used: {len(final_docs)} / 10 max)")
    print(f"  • Total Tokens Used  : {BOLD}{total_tokens}{RESET} tokens (Fixed-10 Baseline: ~{fixed10_tokens_est} tokens)")
    if action == "STOP":
        print(f"  • Token Savings      : {GREEN}{BOLD}Saved ~{tokens_saved_pct:.1f}% tokens{RESET} vs static Fixed-10 RAG")
    print(f"  • Total Query Time   : {total_time:.2f}s (Retrieval: {t_ret1:.2f}s | Gen: {gen_time:.2f}s)")
    print(CYAN + "=" * 76 + RESET + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive-RAG Live Demonstration")
    parser.add_argument("--sample", type=int, choices=[1, 2, 3], help="Pre-select sample question 1, 2, or 3")
    parser.add_argument("--query", type=str, help="Run on a custom natural language query string")
    parser.add_argument("--seed", "-s", type=int, default=None, help="Fixed random seed (omit to get fresh questions each launch)")
    args = parser.parse_args()

    run_demo(selected_sample=args.sample, custom_query=args.query, seed=args.seed)
