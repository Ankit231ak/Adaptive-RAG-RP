"""
Adaptive RAG Pipeline Implementation.
Links Retriever (FAISS/Dense), Evidence State (7 V1 features), Controller (STOP/MORE), and Generator (Ollama/Extractive).
"""
from dataclasses import dataclass, asdict
import os
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)
from logging import Logger

from src.data_and_retrieval import Corpus, Document, DenseRetriever, RetrievedDoc
from src.generation_and_eval import (
    GeneratorWrapper,
    GeneratorOutput,
    normalize_answer,
    exact_match_score,
    f1_score,
    recall_at_k,
    mrr_at_k,
    supporting_fact_coverage
)
from src.method_and_controller import (
    EvidenceFeatures,
    extract_evidence_features,
    validate_evidence_features,
    ControllerRouter
)


@dataclass
class PipelineResult:
    question_id: str
    question: str
    gold_answer: str
    predicted_answer: str
    normalized_gold: str
    normalized_pred: str
    exact_match: bool
    f1: float
    precision: float
    recall: float
    retrieved_count: int
    retrieval_latency: float
    generation_latency: float
    total_latency: float
    input_tokens: int
    output_tokens: int
    action_taken: str
    p_more: float
    evidence_features: Dict[str, float]
    recall_at_retrieved: float
    mrr: float
    supporting_fact_cov: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AdaptiveRAGPipeline:
    """
    Evidence-State Adaptive RAG Pipeline.
    Pipeline:
    Query -> Retrieve Top-5 (D5) -> Extract 7 V1 Evidence Features (X5) ->
    Controller predicts STOP or MORE ->
    If MORE: retrieve documents 6-10 (D6:10) -> Context = D10
    Else: Context = D5 ->
    Generator generates answer -> Logging & Metrics.
    """

    def __init__(
        self,
        corpus: Corpus,
        retriever: Optional[DenseRetriever] = None,
        generator: Optional[GeneratorWrapper] = None,
        router: Optional[ControllerRouter] = None,
        activate_adaptation: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        self.corpus = corpus
        self.config = config or {}
        self.retriever = retriever or DenseRetriever(corpus, config=self.config)
        self.generator = generator or GeneratorWrapper(config=self.config)
        if router is not None:
            self.router = router
        else:
            self.router = ControllerRouter(tau=self.config.get("controller", {}).get("tau", 0.3))
            router_file = os.path.join(self.config.get("paths", {}).get("results_raw_dir", "results/raw"), "controller_router.pkl")
            if os.path.exists(router_file):
                self.router.load(router_file)
                logger.info(f"Loaded trained GBDT controller from {router_file}")
        self.activate_adaptation = activate_adaptation

    def run_single(
        self,
        example: Any,
        initial_k: int = 5,
        maximum_k: int = 10
    ) -> PipelineResult:
        start_pipeline = time.time()
        cands = getattr(example, "context_doc_ids", None)

        # 1. Retrieve initial top-5 and candidate 6-10
        start_ret = time.time()
        d5_ranked, d6_10_ranked = self.retriever.retrieve_split_5_and_5(
            query=example.question,
            candidate_doc_ids=cands
        )
        ret_latency = time.time() - start_ret

        scores_5 = [rd.score for rd in d5_ranked]
        d5_embs = np.array([rd.embedding for rd in d5_ranked if rd.embedding is not None], dtype=np.float32) if d5_ranked else None

        # 2. Extract Evidence-State Features (strictly from top-5)
        query_emb = self.retriever.embedding_model.embed_query(example.question)
        features: EvidenceFeatures = extract_evidence_features(
            scores=scores_5,
            doc_embeddings=d5_embs,
            query_embedding=query_emb
        )

        # Validate numerical features (Day 6 requirement)
        val_checks = validate_evidence_features(features)
        if not val_checks["no_nan"]:
            logger.warning(f"NaN detected in features for question {example.id}")

        # 3. Decision: Controller STOP or MORE
        if self.activate_adaptation:
            action, p_more = self.router.predict_action(features.to_vector())
        else:
            # Day 4 hook without activating adaptation
            action, p_more = "STOP", 0.0

        # 4. Form final context based on decision
        if action == "MORE":
            final_ranked = d5_ranked + d6_10_ranked
        else:
            final_ranked = d5_ranked

        final_docs = [rd.doc for rd in final_ranked]
        final_doc_ids = [rd.doc_id for rd in final_ranked]

        # 5. Generate answer
        gen_out: GeneratorOutput = self.generator.generate(
            question=example.question,
            context_docs=final_docs
        )

        tot_latency = time.time() - start_pipeline

        # 6. Evaluation metrics
        gold_answer = example.answer
        norm_gold = normalize_answer(gold_answer)
        norm_pred = normalize_answer(gen_out.answer)
        em = exact_match_score(gen_out.answer, gold_answer)
        f1, prec, rec = f1_score(gen_out.answer, gold_answer)

        gold_ids = getattr(example, "gold_doc_ids", [])
        supporting_facts = getattr(example, "supporting_facts", [])

        ret_recall = recall_at_k(final_doc_ids, gold_ids, len(final_doc_ids))
        mrr = mrr_at_k(final_doc_ids, gold_ids, len(final_doc_ids))
        sf_cov = supporting_fact_coverage(final_docs, supporting_facts)

        return PipelineResult(
            question_id=example.id,
            question=example.question,
            gold_answer=gold_answer,
            predicted_answer=gen_out.answer,
            normalized_gold=norm_gold,
            normalized_pred=norm_pred,
            exact_match=em,
            f1=round(f1, 4),
            precision=round(prec, 4),
            recall=round(rec, 4),
            retrieved_count=len(final_docs),
            retrieval_latency=round(ret_latency, 4),
            generation_latency=gen_out.latency_sec,
            total_latency=round(tot_latency, 4),
            input_tokens=gen_out.input_tokens,
            output_tokens=gen_out.output_tokens,
            action_taken=action,
            p_more=p_more,
            evidence_features=features.to_dict(),
            recall_at_retrieved=round(ret_recall, 4),
            mrr=round(mrr, 4),
            supporting_fact_cov=round(sf_cov, 4)
        )
