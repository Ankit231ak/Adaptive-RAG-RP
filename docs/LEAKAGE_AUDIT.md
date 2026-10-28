# Data Leakage & Split Audit Report

**Date:** 14 September 2026 (Day 9 Audit Milestone)  
**Status:** PASSED (Zero Leakage Verified)

---

## 1. Audit Objective
To guarantee strict academic integrity and prevent train/validation/test leakage across the Adaptive-RAG pipeline:
1. **Split Exclusivity:** Ensure zero intersection between train, validation, and test question IDs.
2. **Feature Purity (Zero Leakage Rule):** Verify that the 7 post-retrieval evidence-state features extracted from $D_5$ contain no ground truth answers, gold supporting facts, or downstream $F1$ metrics.
3. **Scaler & Model Isolation:** Verify that the feature `StandardScaler` and classifier parameters are fitted exclusively on the 700-query training split.

---

## 2. Split Partition Audit
The 1,000 multi-hop queries from HotpotQA dev distractor were partitioned using stratified deterministic sampling (seed = 42):

| Split | File Path | Total Queries | STOP Labels ($\Delta F1 \le 0.05$) | MORE Labels ($\Delta F1 > 0.05$) | Ratio (STOP:MORE) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Train** | `data/splits/train_ids.json` | 700 | 609 | 91 | 87.0% : 13.0% |
| **Validation** | `data/splits/val_ids.json` | 150 | 131 | 19 | 87.3% : 12.7% |
| **Test (Frozen)** | `data/splits/test_ids.json` | 150 | 130 | 20 | 86.7% : 13.3% |
| **Full Set** | `results/raw/oracle_labels.json` | 1,000 | 870 | 130 | 87.0% : 13.0% |

### Intersection Check:
- $\text{Train} \cap \text{Val} = \emptyset$ (0 overlapping IDs)
- $\text{Train} \cap \text{Test} = \emptyset$ (0 overlapping IDs)
- $\text{Val} \cap \text{Test} = \emptyset$ (0 overlapping IDs)

---

## 3. Evidence-State Feature Purity Audit
All 7 V1 features were audited in `src/method_and_controller/evidence_state.py`:
1. `top1_score`: $s_1$ (FAISS cosine similarity)
2. `mean_score`: $\frac{1}{5}\sum s_i$
3. `score_std`: $\sigma(s_1..s_5)$
4. `margin_top1_top2`: $s_1 - s_2$
5. `score_range`: $s_1 - s_5$
6. `max_similarity`: $\max \cos(\mathbf{e}_q, \mathbf{e}_{d_i})$
7. `mean_redundancy`: $\frac{1}{10}\sum_{i<j} \cos(\mathbf{e}_{d_i}, \mathbf{e}_{d_j})$

### Verification Checks:
- [x] No string matching against `gold_answer`.
- [x] No indexing into `supporting_facts`.
- [x] No access to $F1@5$, $F1@10$, or $\Delta F1$.
- [x] Features computed strictly prior to generating the answer.

---

## 4. Conclusion
The dataset partitioning and feature extraction architecture comply fully with the requirements in Sections 3.4, 4, and 7.1 of the Master Project Plan.
