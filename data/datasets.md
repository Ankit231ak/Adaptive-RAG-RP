# Dataset Specification & Data Pipeline

This document details the dataset specifications, schema definitions, storage structure, and automated preprocessing pipeline used in the **Adaptive RAG** project.

---

## 1. HotpotQA (Primary Dataset)

**Official Page:** [https://hotpotqa.github.io/](https://hotpotqa.github.io/)  
**Hugging Face Source:** [`hotpotqa/hotpot_qa`](https://huggingface.co/datasets/hotpotqa/hotpot_qa) (`distractor` setting)

### Status & Verification
The HotpotQA dataset has been downloaded and formatted into standard HotpotQA JSON schema under `data/hotpotqa/`.

| Dataset Split | Local File Path | File Size | Question Count | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Train Set** | `data/hotpotqa/hotpot_train_v1.1.json` | **619.96 MB** | **90,447** | Downloaded & Verified |
| **Dev (Distractor)** | `data/hotpotqa/hotpot_dev_distractor_v1.json` | **51.22 MB** | **7,405** | Downloaded & Verified |

### Automated Download Command
You can download and format the HotpotQA dataset at any time using:
```bash
python scripts/download_hotpotqa.py
```

---

## 2. Dataset Schema Definition

Each entry in `hotpot_train_v1.1.json` and `hotpot_dev_distractor_v1.json` follows the canonical HotpotQA JSON structure:

```json
{
  "_id": "5a8b57f25542995d1e6f1371",
  "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
  "answer": "yes",
  "type": "comparison",
  "level": "hard",
  "supporting_facts": [
    ["Scott Derrickson", 0],
    ["Ed Wood", 0]
  ],
  "context": [
    [
      "Ed Wood (film)",
      [
        "Ed Wood is a 1994 American biographical period comedy-drama film...",
        "The film concerns the period in Wood's life when he made his best-known films..."
      ]
    ],
    [
      "Scott Derrickson",
      [
        "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.",
        "He lives in Los Angeles, California."
      ]
    ]
  ]
}
```

### Field Breakdown
- `_id` (`str`): Unique identifier for the question.
- `question` (`str`): Natural language multi-hop question.
- `answer` (`str`): Ground-truth short answer.
- `type` (`str`): Multi-hop reasoning type (`comparison` or `bridge`).
- `level` (`str`): Difficulty rating (`easy`, `medium`, `hard`).
- `supporting_facts` (`list[[str, int]]`): Tuples of `[doc_title, sentence_index]` containing gold evidence.
- `context` (`list[[str, list[str]]]`): List of context paragraphs returned for the question.

---

## 3. Stable Document ID Generation

To ensure reproducible indexing and retrieval across runs, every passage in the corpus is assigned a **deterministic, collision-free SHA-256 stable document ID**:

$$\text{doc\_id} = \text{"doc\_"} + \text{SHA256}(\text{title} + "\backslash\text{n}" + \text{text})[:12]$$

### Properties
- **Determinism**: 100% reproducible across machines, OS, and process restarts.
- **Independence**: Immune to array re-ordering or index shifts.
- **Traceability**: Allows direct mapping from supporting facts to exact document passages.

---

## 4. Local Directory Structure & Artifacts

```text
data/
├── hotpotqa/
│   ├── hotpot_train_v1.1.json           # Raw training set (90,447 questions)
│   └── hotpot_dev_distractor_v1.json    # Raw distractor dev set (7,405 questions)
├── processed/
│   └── tiny_corpus.json                 # Processed corpus cache with stable IDs
├── splits/
│   └── tiny_sample_dev.json             # Sample dataset split (20 questions)
└── datasets.md                          # Dataset documentation
```

> **Git Safety Note:**  
> All raw dataset JSON files (`data/**/*.json`) are listed in `.gitignore` to prevent committing heavy binaries to GitHub.

---

## 5. Secondary Dataset: 2WikiMultiHopQA (Planned)

**Dataset Source:** [`DEAD-2/2wikimultihopqa`](https://huggingface.co/datasets/DEAD-2/2wikimultihopqa)

```yaml
dataset: 2WikiMultiHopQA
train_file: train.json
dev_file: dev.json
sample_size: 1000
random_seed: 42
embedding_model: all-MiniLM-L6-v2
initial_k: 5
maximum_k: 10
controller: LogisticRegression
```

---

## 6. Research Pipeline Overview

```text
Dataset (HotpotQA / 2WikiMultiHopQA)
   ↓
Question + Context Passages
   ↓
Corpus Builder & Stable ID Assignment (SHA-256)
   ↓
Retriever (Top-5 Initial Retrieval)
   ↓
Post-Retrieval Evidence State Features (7 V1 Signals)
   ↓
Logistic Regression Controller (Router)
   ↓
┌───────────────────────┐
│                       │
STOP (Use Top-5)    MORE (Retrieve Top-10)
│                       │
↓                       ↓
Generator Wrapper   Generator Wrapper
│                       │
└───────────┬───────────┘
            ↓
  Answer Normalization & EM/F1 Evaluation
```
