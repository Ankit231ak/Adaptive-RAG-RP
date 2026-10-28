import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_and_retrieval.loader import create_sample_dataset
from src.data_and_retrieval.retriever import DenseRetriever
from src.data_and_retrieval.embeddings import OllamaEmbeddingModel
import logging

logging.basicConfig(level=logging.ERROR)

def run_spot_check():
    print("Loading 3 random questions from HotpotQA...")
    questions, corpus = create_sample_dataset(raw_file_path="data/hotpotqa/hotpot_train_v1.1.json", sample_size=3, seed=42)
    
    print("Initializing Qwen Embeddings and FAISS Index...")
    embedding_model = OllamaEmbeddingModel(model_name="qwen3-embedding:4b", dimension=4096)
    retriever = DenseRetriever(corpus=corpus, embedding_model=embedding_model)
    retriever.build_index()

    print("\n" + "="*60)
    for idx, q in enumerate(questions):
        print(f"QUESTION {idx+1}: {q.question}")
        print(f"GOLD ANSWER: {q.answer}")
        print("="*60)
        
        retrieved_docs = retriever.retrieve(q.question, top_k=5, candidate_doc_ids=q.context_doc_ids)
        
        for rank, rd in enumerate(retrieved_docs, start=1):
            print(f"\n  [Rank {rank}] Score: {rd.score:.4f} | Title: {rd.doc.title}")
            snippet = rd.doc.text[:200].replace('\n', ' ')
            print(f"  Text: {snippet}...")
            
        print("\n" + "="*60)

if __name__ == "__main__":
    run_spot_check()
