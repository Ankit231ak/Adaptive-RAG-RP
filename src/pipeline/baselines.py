"""
Standard Baseline Runners (B0 Direct LLM, B1 RAG-3, B2 RAG-5, B3 RAG-10, B4 Simple Score Rule).
Ensures fair comparison under identical generator, retriever, and evaluation protocols.
"""
from dataclasses import dataclass, asdict
import time
from typing import Dict, List, Optional, Any

from src.data_and_retrieval import Corpus, DenseRetriever, RetrievedDoc
from src.generation_and_eval import (
    GeneratorWrapper,
    GeneratorOutput,
    exact_match_score,
    f1_score,
    normalize_answer,
    recall_at_k,
    mrr_at_k,
    supporting_fact_coverage
)


@dataclass
class BaselineResult:
    baseline_id: str
    question_id: str
    question: str
    gold_answer: str
    predicted_answer: str
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
    recall_at_retrieved: float
    mrr: float
    supporting_fact_cov: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaselineRunner:
    """
    Executes standard fixed baselines for fair comparison.
    """

    def __init__(
        self,
        retriever: DenseRetriever,
        generator: GeneratorWrapper,
        config: Optional[Dict[str, Any]] = None
    ):
        self.retriever = retriever
        self.generator = generator
        self.config = config or {}

    def run_direct_llm(self, example: Any) -> BaselineResult:
        """Baseline B0: Direct LLM (k=0, no retrieval)."""
        start_t = time.time()
        gen_out: GeneratorOutput = self.generator.generate_direct(example.question)
        tot_latency = time.time() - start_t

        em = exact_match_score(gen_out.answer, example.answer)
        f1, prec, rec = f1_score(gen_out.answer, example.answer)

        return BaselineResult(
            baseline_id="B0_Direct_LLM",
            question_id=example.id,
            question=example.question,
            gold_answer=example.answer,
            predicted_answer=gen_out.answer,
            exact_match=em,
            f1=round(f1, 4),
            precision=round(prec, 4),
            recall=round(rec, 4),
            retrieved_count=0,
            retrieval_latency=0.0,
            generation_latency=gen_out.latency_sec,
            total_latency=round(tot_latency, 4),
            input_tokens=gen_out.input_tokens,
            output_tokens=gen_out.output_tokens,
            recall_at_retrieved=0.0,
            mrr=0.0,
            supporting_fact_cov=0.0
        )

    def run_fixed_rag(self, example: Any, k: int) -> BaselineResult:
        """Baselines B1 (k=3), B2 (k=5), B3 (k=10): Fixed RAG."""
        start_ret = time.time()
        cands = getattr(example, "context_doc_ids", None)
        ranked_docs: List[RetrievedDoc] = self.retriever.retrieve(
            query=example.question,
            top_k=k,
            candidate_doc_ids=cands
        )
        ret_latency = time.time() - start_ret

        docs = [rd.doc for rd in ranked_docs]
        retrieved_ids = [rd.doc_id for rd in ranked_docs]
        gold_ids = getattr(example, "gold_doc_ids", [])
        supporting_facts = getattr(example, "supporting_facts", [])

        rec_ret = recall_at_k(retrieved_ids, gold_ids, k)
        mrr = mrr_at_k(retrieved_ids, gold_ids, k)
        sf_cov = supporting_fact_coverage(docs, supporting_facts)

        gen_out: GeneratorOutput = self.generator.generate(example.question, docs)
        tot_latency = ret_latency + gen_out.latency_sec

        em = exact_match_score(gen_out.answer, example.answer)
        f1, prec, rec = f1_score(gen_out.answer, example.answer)

        baseline_id = f"B{1 if k == 3 else (2 if k == 5 else 3)}_Fixed_RAG_{k}"

        return BaselineResult(
            baseline_id=baseline_id,
            question_id=example.id,
            question=example.question,
            gold_answer=example.answer,
            predicted_answer=gen_out.answer,
            exact_match=em,
            f1=round(f1, 4),
            precision=round(prec, 4),
            recall=round(rec, 4),
            retrieved_count=len(docs),
            retrieval_latency=round(ret_latency, 4),
            generation_latency=gen_out.latency_sec,
            total_latency=round(tot_latency, 4),
            input_tokens=gen_out.input_tokens,
            output_tokens=gen_out.output_tokens,
            recall_at_retrieved=round(rec_ret, 4),
            mrr=round(mrr, 4),
            supporting_fact_cov=round(sf_cov, 4)
        )

    def run_simple_score_rule(self, example: Any, threshold: float = 0.65) -> BaselineResult:
        """Baseline B4: Top-5 then retrieve next 5 if top score < threshold."""
        start_ret = time.time()
        cands = getattr(example, "context_doc_ids", None)
        d5, d6_10 = self.retriever.retrieve_split_5_and_5(example.question, candidate_doc_ids=cands)
        ret_latency = time.time() - start_ret

        s1 = d5[0].score if d5 else 0.0
        if s1 < threshold:
            final_docs = [rd.doc for rd in d5 + d6_10]
            retrieved_ids = [rd.doc_id for rd in d5 + d6_10]
        else:
            final_docs = [rd.doc for rd in d5]
            retrieved_ids = [rd.doc_id for rd in d5]

        gold_ids = getattr(example, "gold_doc_ids", [])
        supporting_facts = getattr(example, "supporting_facts", [])

        rec_ret = recall_at_k(retrieved_ids, gold_ids, len(final_docs))
        mrr = mrr_at_k(retrieved_ids, gold_ids, len(final_docs))
        sf_cov = supporting_fact_coverage(final_docs, supporting_facts)

        gen_out = self.generator.generate(example.question, final_docs)
        tot_latency = ret_latency + gen_out.latency_sec

        em = exact_match_score(gen_out.answer, example.answer)
        f1, prec, rec = f1_score(gen_out.answer, example.answer)

        return BaselineResult(
            baseline_id="B4_Simple_Score_Rule",
            question_id=example.id,
            question=example.question,
            gold_answer=example.answer,
            predicted_answer=gen_out.answer,
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
            recall_at_retrieved=round(rec_ret, 4),
            mrr=round(mrr, 4),
            supporting_fact_cov=round(sf_cov, 4)
        )

    def run_adaptive_rag_complexity(self, example: Any) -> BaselineResult:
        """Baseline B5: Adaptive-RAG (Jeong et al.) question-complexity routing before retrieval."""
        q_text = example.question.strip()
        tokens = q_text.split()
        q_len = len(tokens)
        q_lower = q_text.lower()

        # Pre-retrieval complexity estimation
        is_comparison = any(kw in q_lower for kw in ["both", "more", "less", "older", "younger", "between", "either", "same"])
        is_chain = any(kw in q_lower for kw in ["where was the", "who was the", "which film", "what team"]) and q_len > 14

        if is_comparison or is_chain or q_len > 18:
            # Complex query -> Multi-hop retrieval (k=10)
            res = self.run_fixed_rag(example, k=10)
        elif q_len <= 7 and not any(kw in q_lower for kw in ["who", "what", "where", "when"]):
            # Simple query -> Direct parametric (k=0)
            res = self.run_direct_llm(example)
        else:
            # Moderate query -> Standard retrieval (k=5)
            res = self.run_fixed_rag(example, k=5)

        res.baseline_id = "B5_Adaptive_RAG_Complexity"
        return res

