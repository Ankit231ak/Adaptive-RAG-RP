"""
HotpotQA raw data loader and sampling routines.
"""
from dataclasses import dataclass, field, asdict
import json
import random
from typing import Dict, List, Optional, Tuple, Any
from src.data_and_retrieval.corpus import Corpus, Document


@dataclass
class SupportingFact:
    """Represents a gold supporting fact."""
    title: str
    sent_idx: int
    doc_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HotpotQAExample:
    """Represents a parsed HotpotQA sample."""
    id: str
    question: str
    answer: str
    type: str
    level: str
    supporting_facts: List[SupportingFact]
    context_doc_ids: List[str]
    gold_doc_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "type": self.type,
            "level": self.level,
            "supporting_facts": [sf.to_dict() for sf in self.supporting_facts],
            "context_doc_ids": self.context_doc_ids,
            "gold_doc_ids": self.gold_doc_ids
        }


def load_raw_hotpotqa(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_hotpotqa_example(raw_ex: Dict[str, Any], corpus: Corpus) -> HotpotQAExample:
    ex_id = raw_ex["_id"]
    question = raw_ex["question"]
    answer = raw_ex["answer"]
    q_type = raw_ex.get("type", "unknown")
    level = raw_ex.get("level", "unknown")

    context_doc_ids = []
    title_to_doc_id = {}
    
    for title, sentences in raw_ex.get("context", []):
        doc = corpus.add_document(title=title, sentences=sentences, metadata={"example_id": ex_id})
        context_doc_ids.append(doc.doc_id)
        title_to_doc_id[title] = doc.doc_id

    supporting_facts = []
    gold_doc_ids_set = set()
    
    for item in raw_ex.get("supporting_facts", []):
        sf_title = item[0]
        sent_idx = item[1]
        doc_id = title_to_doc_id.get(sf_title)
        
        if not doc_id:
            docs = corpus.get_documents_by_title(sf_title)
            if docs:
                doc_id = docs[0].doc_id
        
        sf = SupportingFact(title=sf_title, sent_idx=sent_idx, doc_id=doc_id)
        supporting_facts.append(sf)
        if doc_id:
            gold_doc_ids_set.add(doc_id)

    return HotpotQAExample(
        id=ex_id,
        question=question,
        answer=answer,
        type=q_type,
        level=level,
        supporting_facts=supporting_facts,
        context_doc_ids=context_doc_ids,
        gold_doc_ids=list(gold_doc_ids_set)
    )


def create_sample_dataset(
    raw_file_path: str,
    sample_size: int = 20,
    seed: int = 42
) -> Tuple[List[HotpotQAExample], Corpus]:
    raw_data = load_raw_hotpotqa(raw_file_path)
    rng = random.Random(seed)
    sampled_raw = rng.sample(raw_data, min(sample_size, len(raw_data)))
    
    corpus = Corpus()
    parsed_examples = [parse_hotpotqa_example(raw_ex, corpus) for raw_ex in sampled_raw]
    
    return parsed_examples, corpus
