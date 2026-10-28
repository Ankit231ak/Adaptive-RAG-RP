# V1 Post-Retrieval Evidence-State Feature Dictionary

**Milestone**: Day 6 (10 September 2026)  
**Status**: Frozen for V1 Controller Experimentation  

---

## 1. Overview & Objective

The evidence-state controller operates **strictly after the initial retrieval batch** of top-5 passages ($D_5$). Instead of predicting retrieval necessity solely from the user query $q$, the controller observes the **state of the retrieved evidence** $X_5 = \Phi(q, D_5)$.

The objective is to predict whether retrieving an additional batch of five documents ($D_{6:10}$) will yield a material improvement in generation quality:
$$\Delta F1 = F1(D_{10}) - F1(D_5) > \delta$$

If the current evidence is already high-confidence and comprehensive, the controller outputs **STOP** early, saving latency, retrieval compute, and context-window token costs.

---

## 2. Feature Definitions & Formulas

All 7 V1 features are derived exclusively from the query $q$ and initial top-5 retrieved passages $D_5 = \{d_1, d_2, d_3, d_4, d_5\}$ with associated dense similarity scores $s_1 \ge s_2 \ge s_3 \ge s_4 \ge s_5$.

| Feature Name | Mathematical Formula | Valid Range | Physical / Semantic Interpretation | Extraction Source |
| :--- | :--- | :--- | :--- | :--- |
| **`top1_score`** | $s_1$ | $[-1.0, 1.0]$ | Absolute confidence of the best retrieved evidence passage. Higher values indicate strong initial match. | First ranked retrieval score from FAISS. |
| **`mean_score`** | $\frac{1}{5} \sum_{i=1}^5 s_i$ | $[-1.0, 1.0]$ | Average query-evidence alignment across the top-5 candidate passages. | Mean of top-5 retrieval scores. |
| **`score_std`** | $\sqrt{\frac{1}{5} \sum_{i=1}^5 (s_i - \bar{s})^2}$ | $[0.0, 1.0]$ | Dispersion of retrieval scores. Low std indicates uniform relevance or noise; high std indicates sharp distinction between relevant and irrelevant docs. | Population standard deviation of top-5 retrieval scores. |
| **`margin_top1_top2`** | $s_1 - s_2$ | $[0.0, 2.0]$ | Retrieval ranking confidence. A wide margin indicates the top document stands out distinctly. A near-zero margin indicates competition between top documents. | Difference between 1st and 2nd scores. |
| **`score_range`** | $s_1 - s_5$ | $[0.0, 2.0]$ | Evidence cliff drop. Measures how quickly retrieval quality degrades from rank 1 to rank 5. | Difference between 1st and 5th scores. |
| **`max_similarity`** | $\max_{i \in \{1\dots 5\}} \cos(\mathbf{e}_q, \mathbf{e}_{d_i})$ | $[-1.0, 1.0]$ | Maximum semantic cosine similarity between query embedding and evidence embeddings. | Vector inner product between normalized embeddings. |
| **`mean_redundancy`** | $\frac{1}{10} \sum_{1 \le i < j \le 5} \cos(\mathbf{e}_{d_i}, \mathbf{e}_{d_j})$ | $[-1.0, 1.0]$ | Pairwise inter-document redundancy. High redundancy indicates retrieved passages duplicate information; low redundancy indicates multi-hop coverage or scattered topics. | Mean of upper-triangle cosine similarity matrix of $D_5$ embeddings. |

---

## 3. Strict Academic Integrity & Leakage Prohibition

In accordance with Section 4 and Section 16 of the Master Project Plan:

> [!CAUTION]
> **Zero Leakage Rule**: At inference time, the gold answer, gold document IDs, and gold supporting facts are completely unavailable.

The feature extraction pipeline strictly prohibits and audits against:
1. **No Gold Answer Overlap**: No string or lexical overlap with $y_{\text{gold}}$.
2. **No Gold Supporting Facts**: No indicators of whether a retrieved document is a true supporting fact.
3. **No Downstream F1 / Evaluation Metric**: Neither $F1@5$, $F1@10$, nor $\Delta F1$ may ever enter the controller feature vector $X_5$.
4. **No LLM Output Signals**: Features must be computed **before** generation, using only $q$ and $D_5$.

---

## 4. Empirical Distribution Validation (Day 6 Audit, N = 60)

From the automated audit conducted on 10 September (`scripts/extract_features.py`):

| Feature Name | Mean | Std | Min | Median | Max | Audit Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `top1_score` | 0.6639 | 0.0953 | 0.4629 | 0.6652 | 0.8575 | Verified In Bounds |
| `mean_score` | 0.5047 | 0.0949 | 0.3210 | 0.5101 | 0.7000 | Monotonically Below Top-1 |
| `score_std` | 0.0958 | 0.0454 | 0.0161 | 0.0930 | 0.2256 | Non-negative ($\ge 0$) |
| `margin_top1_top2` | 0.1279 | 0.1088 | 0.0068 | 0.0961 | 0.4347 | Non-negative ($\ge 0$) |
| `score_range` | 0.2596 | 0.1213 | 0.0479 | 0.2495 | 0.6286 | Non-negative ($\ge 0$) |
| `max_similarity` | 0.6639 | 0.0953 | 0.4629 | 0.6652 | 0.8575 | Verified In Bounds |
| `mean_redundancy` | 0.4272 | 0.1374 | 0.1578 | 0.4444 | 0.7292 | Verified In $[-1, 1]$ |

**Audit Conclusion**: Zero validation failures (100% pass rate). Features are numerically well-conditioned and ready for controller training.
