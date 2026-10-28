"""
Corpus representation and stable document ID management.
"""
from dataclasses import dataclass, field, asdict
import hashlib
from typing import Dict, List, Optional, Any


def generate_stable_doc_id(title: str, text: str, prefix: str = "doc_", length: int = 12) -> str:
    """Generate deterministic SHA-256 stable document ID."""
    content = f"{title.strip()}\n{text.strip()}"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}{digest}"


@dataclass
class Document:
    """Represents a single document/passage in the retrieval corpus."""
    doc_id: str
    title: str
    text: str
    sentences: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        return cls(**data)


class Corpus:
    """In-memory document corpus with stable lookup."""
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.title_to_doc_ids: Dict[str, List[str]] = {}

    def add_document(self, title: str, sentences: List[str], metadata: Optional[Dict[str, Any]] = None) -> Document:
        text = " ".join(s.strip() for s in sentences if s.strip())
        doc_id = generate_stable_doc_id(title, text)
        
        if doc_id not in self.documents:
            doc = Document(
                doc_id=doc_id,
                title=title,
                text=text,
                sentences=sentences,
                metadata=metadata or {}
            )
            self.documents[doc_id] = doc
            if title not in self.title_to_doc_ids:
                self.title_to_doc_ids[title] = []
            if doc_id not in self.title_to_doc_ids[title]:
                self.title_to_doc_ids[title].append(doc_id)
        
        return self.documents[doc_id]

    def get_document(self, doc_id: str) -> Optional[Document]:
        return self.documents.get(doc_id)

    def get_documents_by_title(self, title: str) -> List[Document]:
        doc_ids = self.title_to_doc_ids.get(title, [])
        return [self.documents[did] for did in doc_ids]

    def __len__(self) -> int:
        return len(self.documents)

    def to_list(self) -> List[Dict[str, Any]]:
        return [doc.to_dict() for doc in self.documents.values()]

    @classmethod
    def from_list(cls, doc_list: List[Dict[str, Any]]) -> "Corpus":
        corpus = cls()
        for d in doc_list:
            doc = Document.from_dict(d)
            corpus.documents[doc.doc_id] = doc
            if doc.title not in corpus.title_to_doc_ids:
                corpus.title_to_doc_ids[doc.title] = []
            corpus.title_to_doc_ids[doc.title].append(doc.doc_id)
        return corpus
