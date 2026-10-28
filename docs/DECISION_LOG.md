# Adaptive RAG — Project Decision Log & Ownership Register

## 1. Fixed Team Role Assignments

| Member | Name | Primary Role | Owned Modules & Components |
| :--- | :--- | :--- | :--- |
| **Member A** | Member A | Method & Controller | `src/router.py`, `src/evidence_state.py`, `src/oracle.py`, `src/adaptive_rag.py`, `docs/FEATURE_DICTIONARY.md` |
| **Member B** | Member B | Data & Retrieval | `src/data.py`, `src/corpus.py`, `src/retriever.py`, `src/embeddings.py`, `scripts/build_index.py`, `scripts/download_hotpotqa.py` |
| **Member C** | Member C | Generation & Evaluation | `src/generator.py`, `src/metrics.py`, `src/baselines.py`, `scripts/run_baselines.py`, `scripts/extract_features.py`, `main.py` |

---

## 2. Core Architecture Decisions (Days 1 – 6)

### Decision 1: Answer Normalization & Metrics (Day 1 - 5 Sep)
- **Decision**: Adopt official HotpotQA evaluation metrics (lowercasing, punctuation stripping, article removal, token-level F1 and EM match).
- **Status**: Verified in `src/metrics.py`.

### Decision 2: Fixed Prompt Template (Day 1 - 5 Sep)
- **Decision**: Implement a frozen instruction-tuned prompt template formatting retrieved passages concisely:
  ```text
  SYSTEM: You are an accurate question-answering assistant.
  CONTEXT: Passage [1] (Title): Text ...
  QUESTION: ...
  ANSWER:
  ```
- **Status**: Implemented in `src/generator.py`.

### Decision 3: Deterministic Stable Document IDs (Day 1 - 5 Sep)
- **Decision**: Generate document IDs using SHA-256 hash:
  $$\text{doc\_id} = \text{"doc\_"} + \text{SHA256}(\text{title} + "\backslash\text{n}" + \text{text})[:12]$$
- **Status**: Implemented in `src/corpus.py`.

### Decision 4: V1 Controller State and Action Space (Day 2 - 6 Sep)
- **Decision**: Define state as the 7-dimensional post-retrieval evidence-state vector $X_5$ extracted strictly from top-5 retrieved passages. Action space is binary: $\{\text{STOP}, \text{MORE}\}$.
- **Status**: Implemented in `src/evidence_state.py` and `src/router.py`.

### Decision 5: FAISS Dense Retrieval & Embeddings Architecture (Day 3 - 7 Sep)
- **Decision**: Implement FAISS `IndexFlatIP` (inner product on L2-normalized embeddings == cosine similarity). Integrate support for Ollama embeddings with configurable space for model name (e.g. `nomic-embed-text`) alongside local `SentenceTransformer` (`all-MiniLM-L6-v2`) and NumPy fallback.
- **Status**: Implemented in `src/retriever.py` and `src/embeddings.py`. Verified via `scripts/build_index.py`.

### Decision 6: Ollama LLM Generator & Graceful Fallback (Day 3 - 7 Sep)
- **Decision**: Implement `OllamaClient` in `src/generator.py` querying local Ollama instance (`http://localhost:11434/api/generate`). Provide an explicit configuration space for the user's Ollama model name, while supporting extractive baseline fallback when the model name is unset or Ollama is offline.
- **Status**: Implemented in `src/generator.py` and tested across baselines.

### Decision 7: Fixed-k RAG Baselines & 5+5 Non-Duplicate Splitting (Day 4 - 8 Sep)
- **Decision**: Support fixed $k \in \{3, 5, 10\}$ from a single index. Guarantee that $k=10$ retrieval decomposes cleanly into initial $D_5$ and expansion $D_{6:10}$ with zero duplicate IDs.
- **Status**: Implemented in `DenseRetriever.retrieve_split_5_and_5()` and `src/baselines.py`.

### Decision 8: Direct LLM Baseline & Retrieval Correctness Auditing (Day 5 - 9 Sep)
- **Decision**: Implement Direct LLM ($k=0$) baseline to measure parametric memory versus retrieval necessity. Implement retrieval evaluation metrics (Recall@3, Recall@5, Recall@10, MRR).
- **Status**: Implemented in `src/metrics.py` and evaluated via `scripts/run_baselines.py`.

### Decision 9: The 7 V1 Evidence-State Features & Zero-Leakage Guarantee (Day 6 - 10 Sep)
- **Decision**: Implement all 7 V1 evidence features:
  1. `top1_score` ($s_1$)
  2. `mean_score` ($\bar{s}_{1..5}$)
  3. `score_std` ($\sigma_{1..5}$)
  4. `margin_top1_top2` ($s_1 - s_2$)
  5. `score_range` ($s_1 - s_5$)
  6. `max_similarity` ($\max \cos(q, d_i)$)
  7. `mean_redundancy` ($\text{mean}_{i < j} \cos(d_i, d_j)$)
  Enforce strict leakage prohibition: no gold answers, no gold supporting facts, and no future F1 can be accessed by the feature extractor.
- **Status**: Implemented in `src/evidence_state.py`, documented in `docs/FEATURE_DICTIONARY.md`, and validated over 60 queries with 0% failure rate in `scripts/extract_features.py`.

---

## 3. Daily Milestones Completion Register (Days 1 to 6)

| Date | Milestone | Evidence / Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **5 Sep (Day 1)** | Kickoff & runnable skeleton | Method spec & state machine | Tiny HotpotQA sample & stable IDs | Fixed prompt & EM/F1 skeleton | `results/raw/hello_world_run.json` |
| **6 Sep (Day 2)** | Freeze V1 controller spec & literature | STOP/MORE interface spec | Dense retriever + FAISS plan | Prompt parser & answer types | `docs/READING_NOTES.md` (Adaptive-RAG + Self-RAG) |
| **7 Sep (Day 3)** | Build retrieval & connect generation | Evidence-state feature signature | FAISS index + dense embeddings | Connect retrieved text to LLM | `scripts/build_index.py`, `faiss_index.bin` |
| **8 Sep (Day 4)** | Standard RAG baseline (fixed-k) | Adaptive interface hooks | Fixed $k \in \{3,5,10\}$ on same index | Standard RAG answers across $k$ | `results/raw/b1_fixed_rag_3_outputs.json`, etc. |
| **9 Sep (Day 5)** | Baseline quality & literature | Baseline delta inspection | Retrieval Recall@k & MRR checks | Direct LLM ($k=0$) & EM/F1 suite | `results/tables/baselines_summary.csv`, `docs/READING_NOTES.md` |
| **10 Sep (Day 6)** | Build evidence-state dataset | 7 V1 features (zero leakage) | Validate score distributions & redundancy | Feature logger & CSV/JSONL output | `results/raw/evidence_features_day6.jsonl`, `docs/FEATURE_DICTIONARY.md` |

---

## 4. Verification Sign-Off

- [x] **All 11 automated unit and integration tests passed** (`scripts/verify_day6_milestones.py`).
- [x] **FAISS vector store and dense retrieval verified** (`scripts/build_index.py`).
- [x] **Baselines suite (B0, B1, B2, B3, B4) runnable and saved** (`scripts/run_baselines.py`).
- [x] **Day 6 feature extraction and validation verified on 60 queries** (`scripts/extract_features.py`).
- [x] **Zero data leakage verified**: no test IDs, gold answers, or downstream metrics inside features.
- [x] **Ollama model configuration space created** in `configs/base.yaml`, ready for user's model specification.
