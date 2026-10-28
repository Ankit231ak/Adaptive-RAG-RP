# Improving Adaptive Retrieval-Augmented Generation (RAG) for Multi-Hop Question Answering

[![CI](https://img.shields.io/badge/CI-Passing-brightgreen?logo=github-actions&logoColor=white)](.github/workflows/ci.yml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)

This repository contains the official implementation of the research paper:  
**"Improving Adaptive Retrieval-Augmented Generation (RAG) for Multi-Hop Question Answering"**.

The framework introduces an **Evidence-State Marginal Retrieval Controller**, a lightweight post-retrieval routing mechanism that inspects an initial retrieval batch ($D_5$), extracts seven statistical evidence-state features ($X_5 \in \mathbb{R}^7$), and dynamically decides whether to early-exit (**STOP**) or retrieve an additional batch of bridging documents (**MORE**, $k=10$).

---

## 1. Key Benchmark Results

Evaluated across 1,000 multi-hop HotpotQA queries under a strict frozen train/validation/test split (700 Train / 150 Val / 150 Test):

### Frozen Test Split ($N=150$ Unseen Multi-Hop Queries)

| Method | Baseline ID | Exact Match (EM) | F1 Score | Avg Docs | Avg Tokens | Latency | Early STOP Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Direct LLM (No Retrieval) | B0 | 12.00% | 0.2187 | 0.00 | 98.2 | 0.41s | 100.0% |
| Fixed RAG-3 | B1 | 36.67% | 0.5394 | 3.00 | 480.7 | 1.37s | — |
| Fixed RAG-5 (Initial Budget) | B2 | 34.67% | 0.5443 | 5.00 | 761.8 | 1.67s | — |
| Fixed RAG-10 (Over-retrieval) | B3 | 36.00% | 0.5417 | 10.00 | 1523.7 | 2.39s | — |
| Simple Score Rule | B4 | 36.00% | 0.5387 | 7.10 | 1076.6 | 1.93s | 58.0% |
| Adaptive-RAG (Jeong et al.) | B5 | 34.67% | 0.5249 | 6.60 | 1003.6 | 1.92s | 68.0% |
| **Proposed Adaptive-RAG (Ours)** | **B6** | **36.00%** | **0.5560** | **5.23** | **796.1** | **1.70s** | **95.3%** |
| *Oracle Upper Bound (Hindsight)* | *E5* | *40.67%* | *0.6140* | *5.73* | *865.8* | *1.78s* | *85.3%* |

### Key Empirical Findings:
1. **Mitigating Context Bloat**: Expanding retrieval blindly from $k=5$ to $k=10$ degrades F1 score from 0.5443 down to 0.5417 due to semantic distractors, while doubling token consumption (761.8 $\to$ 1523.7).
2. **Quality Win**: The proposed controller achieves the **highest F1 score (0.5560)** among all methods, outperforming Fixed RAG-10, Fixed RAG-5, and question-complexity routing.
3. **Efficiency Gains**: The model early-exits on **95.3% of queries**, reducing input token usage by **48%** and cutting generation latency by **29%**.

---

## 2. Methodology & Mathematical Formulation

### 2.1 Post-Retrieval Evidence-State Features ($X_5 \in \mathbb{R}^7$)
Computed strictly from the user query $q$ and top-5 retrieved passages $D_5 = \{d_1, \dots, d_5\}$ with dense similarity scores $s_1 \ge s_2 \ge \dots \ge s_5$:

1. **Top-1 Score**: $s_1 = \max_i s_i$ (absolute confidence of top document).
2. **Mean Score**: $\bar{s} = \frac{1}{5}\sum_{i=1}^5 s_i$ (average query-evidence alignment).
3. **Score Dispersion**: $\sigma_s = \sqrt{\frac{1}{5}\sum_{i=1}^5 (s_i - \bar{s})^2}$ (retrieval confidence spread).
4. **Top-1 to Top-2 Margin**: $\Delta s_{12} = s_1 - s_2$ (ranking decisiveness).
5. **Score Range**: $s_1 - s_5$ (evidence degradation cliff).
6. **Max Similarity**: $\max_i \cos(\mathbf{e}_q, \mathbf{e}_{d_i})$ (maximum semantic match).
7. **Mean Pairwise Redundancy**: $\frac{1}{10}\sum_{1 \le i < j \le 5} \cos(\mathbf{e}_{d_i}, \mathbf{e}_{d_j})$ (inter-document diversity).

### 2.2 Offline Marginal-Gain Supervision Oracle
During training, candidate answers $a_5$ and $a_{10}$ are generated offline. Ground-truth label $y \in \{\text{STOP}, \text{MORE}\}$ is assigned as:
$$y = \begin{cases} \text{MORE} & \text{if } F1(a_{10}, y^*) - F1(a_5, y^*) > \delta \\ \text{STOP} & \text{otherwise} \end{cases}$$
where $\delta = 0.05$ denotes the material improvement threshold. In HotpotQA, this yields an **87.0% STOP : 13.0% MORE** class distribution.

### 2.3 Deep Neural Controller Architecture (`NeuralEvidenceRouter`)
To address the severe 87:13 class imbalance and capture non-linear feature interactions, we provide a PyTorch Deep Residual Router:
- **Input Projection**: $\text{Linear}(7 \to 64) \to \text{LayerNorm} \to \text{SiLU}$
- **Residual Blocks**: Stacks of pre-activation residual blocks:
  $$x_{l+1} = x_l + \text{LayerNorm}\Big(\text{Linear}\big(\text{Dropout}(\text{SiLU}(\text{Linear}(\text{LayerNorm}(x_l)))))\big)\Big)$$
- **Focal Loss Objective**:
  $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
  with $\alpha = 0.85$ (weighting the minority MORE class) and focusing parameter $\gamma = 2.0$.
- **Decision Rule**:
  $$\text{Action} = \begin{cases} \text{MORE} & \text{if } P(\text{MORE} \mid X_5) \ge \tau \\ \text{STOP} & \text{if } P(\text{MORE} \mid X_5) < \tau \end{cases}$$
  where $\tau = 0.3$ is calibrated on the validation set.

---

## 3. Repository Structure

```
├── configs/
│   ├── base.yaml              # Centralized hyperparameters, model names, paths
│   ├── retrieval.yaml         # FAISS vector store and embedding configurations
│   └── experiments.yaml       # Baselines B0-B4 definitions and evaluation metrics
├── data/
│   ├── splits/                # Frozen train (700), val (150), test (150) query IDs
│   └── processed/             # FAISS index and vector representations
├── docs/
│   ├── RESEARCH_PAPER.md      # Full academic manuscript and theoretical formulation
│   ├── FEATURE_DICTIONARY.md  # Detailed feature specifications and value domains
│   ├── LEAKAGE_AUDIT.md       # Audit certifying zero train/val/test data leakage
│   └── DISTRACTOR_EXAMPLES.md # Case studies on k=10 distractor degradations
├── figures/                   # Pareto frontier plots and decision confusion matrices
├── main.py                    # Master end-to-end pipeline entry point
├── results/
│   ├── raw/                   # Raw inference logs and controller checkpoints (.pkl)
│   └── tables/                # Aggregated CSV evaluation tables (B0-B6, ablations)
├── scripts/
│   ├── train_controller.py    # Zero-leakage trainer for Neural & GBDT controllers
│   ├── tune_controller.py     # Grid search & validation threshold calibrator
│   ├── run_baselines.py       # Fixed-k baseline runner (B0, B1, B2, B3, B4)
│   ├── evaluate_adaptive.py   # Test-time adaptive pipeline evaluation
│   ├── evaluate_all_methods.py# Master evaluator across all baselines and splits
│   ├── run_ablations.py       # Runs essential feature/architecture ablations (A0-A6)
│   └── make_figures.py        # Generates publication-ready Pareto frontier figures
├── src/
│   ├── data_and_retrieval/    # FAISS vector store, corpus hash IDs, dense embeddings
│   ├── generation_and_eval/   # LLM generator wrapper, prompt formatting, EM/F1 metrics
│   ├── method_and_controller/ # 7 V1 features, Oracle labeler, Neural & GBDT routers
│   └── pipeline/              # AdaptiveRAGPipeline and BaselineRunner orchestrators
└── tests/
    └── test_model_diagnostics.py # Tensor dimensions, forward/backward pass, gradient checks
```

---

## 4. Quickstart & Execution Guide

### 4.1 Prerequisites
```bash
pip install -r requirements.txt
```

### 4.2 Run Diagnostic Verification
Verify tensor dimensions, forward/backward pass stability, and gradient flow:
```bash
python -m unittest tests/test_model_diagnostics.py
```

### 4.3 End-to-End Pipeline Demo
Execute the full Adaptive RAG pipeline on sample multi-hop queries:
```bash
python main.py --samples 20 --seed 42
```

### 4.4 Train the Controller (Zero Leakage)
Train strictly on the 700-query training split with validation calibration:
```bash
# PyTorch Neural Residual Controller (Default)
python scripts/train_controller.py --backend neural --epochs 50 --tau 0.3

# GBDT Decision Tree Controller
python scripts/train_controller.py --backend gbdt --tau 0.3
```

### 4.5 Evaluate All Methods on Frozen Splits
Generate the comprehensive leaderboard across all methods (B0 – B6) on the unseen test split:
```bash
python scripts/evaluate_all_methods.py
```

### 4.6 Run Ablations & Sensitivity Sweeps
Evaluate feature importances, linear vs tree vs neural architectures, and $\tau$ thresholds:
```bash
python scripts/run_ablations.py
```

---

## 5. Strict Zero-Leakage Guarantee

As verified in [`docs/LEAKAGE_AUDIT.md`](docs/LEAKAGE_AUDIT.md):
- $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
- Feature extraction ($X_5$) uses only query $q$ and passages $D_5$ before generation.
- No ground truth answers, gold supporting facts, or downstream $F1$ metrics enter the controller.
- Feature scaling (`StandardScaler`) is fitted strictly on the 700-sample training partition.

---

## 6. Citation

```bibtex
@article{adaptiverag2026,
  title={Improving Adaptive Retrieval-Augmented Generation (RAG) for Multi-Hop Question Answering},
  author={Adaptive RAG Research Team},
  year={2026}
}
```
