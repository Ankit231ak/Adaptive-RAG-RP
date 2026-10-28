"""
Master Verification Script for all deliverables up to 10 September (Days 1 - 6).
"""
import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import load_config, set_seed
from src import (
    Corpus,
    Document,
    generate_stable_doc_id,
    create_sample_dataset,
    DenseRetriever,
    FAISSVectorStore,
    RetrievedDoc
)
from src import (
    GeneratorWrapper,
    GeneratorOutput,
    OllamaClient,
    exact_match_score,
    f1_score,
    normalize_answer,
    recall_at_k,
    mrr_at_k,
    supporting_fact_coverage,
    evaluate_retrieval
)
from src import (
    EvidenceFeatures,
    extract_evidence_features,
    validate_evidence_features,
    compute_oracle_label,
    create_oracle_record,
    ControllerRouter,
    FEATURE_NAMES
)
from src.pipeline import AdaptiveRAGPipeline, BaselineRunner


class TestDay1To6Milestones(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config("configs/base.yaml")
        set_seed(cls.config.get("seed", 42))
        cls.raw_path = cls.config["paths"]["raw_dev_data"]
        cls.examples, cls.corpus = create_sample_dataset(cls.raw_path, sample_size=20, seed=42)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def test_member_b_stable_doc_id_determinism(self):
        """Day 1 & Day 2: Verify deterministic SHA-256 document IDs."""
        id1 = generate_stable_doc_id("Albert Einstein", "Physicist who developed relativity.")
        id2 = generate_stable_doc_id("Albert Einstein", "Physicist who developed relativity.")
        self.assertEqual(id1, id2)
        self.assertTrue(id1.startswith("doc_"))
        self.assertEqual(len(id1), 4 + 12)

    def test_member_b_faiss_vector_store_and_embeddings(self):
        """Day 3: Verify dense vector store indexing and inner product search."""
        store = FAISSVectorStore(dimension=4)
        vectors = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.7071, 0.7071, 0.0, 0.0]
        ], dtype=np.float32)
        doc_ids = ["doc_1", "doc_2", "doc_3"]
        store.add(vectors, doc_ids)

        q_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(q_vec, top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "doc_1")
        self.assertAlmostEqual(results[0][1], 1.0, places=3)

    def test_member_b_fixed_k_and_split_5_and_5(self):
        """Day 4: Verify fixed-k (3, 5, 10) and clean 5+5 split without duplicates."""
        retriever = DenseRetriever(corpus=self.corpus, config=self.config)
        retriever.build_index()

        ex = self.examples[0]
        cands = ex.context_doc_ids

        res_k3 = retriever.retrieve(ex.question, top_k=3, candidate_doc_ids=cands)
        res_k5 = retriever.retrieve(ex.question, top_k=5, candidate_doc_ids=cands)
        res_k10 = retriever.retrieve(ex.question, top_k=10, candidate_doc_ids=cands)

        self.assertEqual(len(res_k3), 3)
        self.assertEqual(len(res_k5), 5)
        self.assertEqual(len(res_k10), 10)

        # Monotonic score check
        scores = [r.score for r in res_k10]
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i], scores[i+1] - 1e-5)

        # 5+5 split
        d5, d6_10 = retriever.retrieve_split_5_and_5(ex.question, candidate_doc_ids=cands)
        self.assertEqual(len(d5), 5)
        self.assertEqual(len(d6_10), 5)
        d5_ids = set(r.doc_id for r in d5)
        d6_10_ids = set(r.doc_id for r in d6_10)
        self.assertEqual(len(d5_ids.intersection(d6_10_ids)), 0, "No duplicate IDs between D5 and D6:10")

    def test_member_b_retrieval_correctness_metrics(self):
        """Day 5: Verify Recall@k and MRR computation."""
        ret_ids = ["doc_a", "doc_b", "doc_c", "doc_d", "doc_e"]
        gold_ids = ["doc_b", "doc_d"]

        rec3 = recall_at_k(ret_ids, gold_ids, k=3)
        self.assertEqual(rec3, 0.5)  # only doc_b in top 3
        rec5 = recall_at_k(ret_ids, gold_ids, k=5)
        self.assertEqual(rec5, 1.0)  # both doc_b and doc_d in top 5

        mrr = mrr_at_k(ret_ids, gold_ids, k=5)
        self.assertEqual(mrr, 0.5)  # first gold doc_b is at rank 2 -> 1/2

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def test_member_c_generator_and_ollama_space(self):
        """Day 1 & Day 3: Verify GeneratorWrapper and space for Ollama model."""
        # Unconfigured Ollama model gracefully uses extractive fallback
        gen = GeneratorWrapper(config={"generator": {"model_name": ""}})
        self.assertEqual(gen.model_name, "")
        out: GeneratorOutput = gen.generate("What is the capital?", [self.corpus.documents[list(self.corpus.documents.keys())[0]]])
        self.assertIsInstance(out.answer, str)
        self.assertGreater(out.latency_sec, 0.0)
        self.assertGreater(out.input_tokens, 0)

    def test_member_c_direct_llm_baseline(self):
        """Day 5: Direct LLM baseline (k=0) generation."""
        gen = GeneratorWrapper(config=self.config)
        out = gen.generate_direct("What year was the university founded?")
        self.assertEqual(out.retrieval_k, 0)
        self.assertIsInstance(out.answer, str)

    def test_member_c_answer_normalization_and_f1(self):
        """Day 1 & Day 5: HotpotQA official normalization and F1 scoring."""
        self.assertEqual(normalize_answer("The University of Oxford."), "university of oxford")
        self.assertTrue(exact_match_score("Oxford", "the Oxford"))
        f1, p, r = f1_score("University of Cambridge", "University of Cambridge, UK")
        self.assertGreater(f1, 0.6)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def test_member_a_seven_v1_features_extraction(self):
        """Day 6: Verify all 7 V1 features extracted with strict leakage prohibition."""
        scores = [0.92, 0.85, 0.78, 0.70, 0.60]
        mock_embs = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.7, 0.3, 0.0],
            [0.6, 0.4, 0.0]
        ], dtype=np.float32)
        q_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        feat: EvidenceFeatures = extract_evidence_features(scores, mock_embs, q_emb)

        self.assertEqual(feat.top1_score, 0.92)
        self.assertAlmostEqual(feat.margin_top1_top2, 0.07, places=4)
        self.assertAlmostEqual(feat.score_range, 0.32, places=4)
        self.assertGreater(feat.mean_redundancy, 0.0)
        self.assertLessEqual(feat.mean_redundancy, 1.0)

        vec = feat.to_vector()
        self.assertEqual(len(vec), 7)
        self.assertEqual(len(FEATURE_NAMES), 7)

        val_checks = validate_evidence_features(feat)
        self.assertTrue(all(val_checks.values()), f"Validation failed: {val_checks}")

    def test_member_a_oracle_logic(self):
        """Day 6: Test delta-F1 oracle computation."""
        # Case 1: significant gain (MORE)
        label1, delta1 = compute_oracle_label(f1_d5=0.40, f1_d10=0.60, delta_threshold=0.05)
        self.assertEqual(label1, "MORE")
        self.assertAlmostEqual(delta1, 0.20, places=4)

        # Case 2: negligible gain (STOP)
        label2, delta2 = compute_oracle_label(f1_d5=0.70, f1_d10=0.72, delta_threshold=0.05)
        self.assertEqual(label2, "STOP")
        self.assertAlmostEqual(delta2, 0.02, places=4)

    def test_member_a_controller_router_train_and_predict(self):
        """Day 6: ControllerRouter binary classification and threshold."""
        router = ControllerRouter(tau=0.5)
        # Synthetic training data
        X = [
            [0.95, 0.90, 0.02, 0.05, 0.10, 0.95, 0.85],  # saturated -> STOP
            [0.92, 0.88, 0.03, 0.04, 0.12, 0.92, 0.80],  # saturated -> STOP
            [0.55, 0.40, 0.15, 0.02, 0.35, 0.55, 0.20],  # poor/diverse -> MORE
            [0.50, 0.35, 0.18, 0.01, 0.38, 0.50, 0.18],  # poor/diverse -> MORE
        ]
        y = ["STOP", "STOP", "MORE", "MORE"]
        router.train(X, y)
        self.assertTrue(router.is_trained)

        action, p_more = router.predict_action([0.94, 0.89, 0.02, 0.05, 0.11, 0.94, 0.82])
        self.assertEqual(action, "STOP")

    # -------------------------------------------------------------------------
    # Team End-to-End Integration Pipeline Tests
    # -------------------------------------------------------------------------

    def test_team_end_to_end_adaptive_rag_pipeline(self):
        """Day 3 & Day 4: End-to-end pipeline execution."""
        retriever = DenseRetriever(corpus=self.corpus, config=self.config)
        retriever.build_index()
        pipeline = AdaptiveRAGPipeline(corpus=self.corpus, retriever=retriever, config=self.config)

        res = pipeline.run_single(self.examples[0])
        self.assertIn(res.action_taken, ["STOP", "MORE"])
        self.assertIn(res.retrieved_count, [5, 10])
        self.assertGreater(res.total_latency, 0.0)
        self.assertEqual(len(res.evidence_features), 7)


if __name__ == "__main__":
    unittest.main()
