"""
Adaptive RAG Package — Modular Architecture
"""
from src.common import load_config, set_seed
from src.data_and_retrieval import (
    Corpus,
    Document,
    generate_stable_doc_id,
    Retriever,
    DenseRetriever,
    FAISSVectorStore,
    RetrievedDoc,
    create_sample_dataset
)
from src.generation_and_eval import (
    GeneratorWrapper,
    GeneratorOutput,
    OllamaClient,
    evaluate_predictions,
    normalize_answer,
    exact_match_score,
    f1_score,
    recall_at_k,
    mrr_at_k,
    supporting_fact_coverage,
    evaluate_retrieval
)
from src.method_and_controller import (
    ControllerRouter,
    EvidenceFeatures,
    extract_evidence_features,
    validate_evidence_features,
    create_oracle_record,
    compute_oracle_label,
    FEATURE_NAMES
)
from src.pipeline import AdaptiveRAGPipeline, PipelineResult

__all__ = [
    "load_config",
    "set_seed",
    "Corpus",
    "Document",
    "generate_stable_doc_id",
    "Retriever",
    "DenseRetriever",
    "FAISSVectorStore",
    "RetrievedDoc",
    "create_sample_dataset",
    "GeneratorWrapper",
    "GeneratorOutput",
    "OllamaClient",
    "evaluate_predictions",
    "normalize_answer",
    "exact_match_score",
    "f1_score",
    "recall_at_k",
    "mrr_at_k",
    "supporting_fact_coverage",
    "evaluate_retrieval",
    "ControllerRouter",
    "EvidenceFeatures",
    "extract_evidence_features",
    "validate_evidence_features",
    "create_oracle_record",
    "compute_oracle_label",
    "FEATURE_NAMES",
    "AdaptiveRAGPipeline",
    "PipelineResult"
]

