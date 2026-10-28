# Evidence-State Marginal Retrieval Controller for Adaptive RAG

**Deepak** (Method & Controller), **Ankit** (Data & Retrieval), **Utsav** (Generation & Evaluation)  
*Adaptive RAG Master Research Project*

---

## Abstract
Standard Retrieval-Augmented Generation (RAG) systems operate with a static retrieval budget ($k$), indiscriminately retrieving the same number of documents regardless of query complexity or initial retrieval quality. In multi-hop question answering, retrieving a large fixed batch ($k=10$) often doubles token costs and introduces semantic distractors that induce factual hallucinations, while an overly conservative budget ($k=5$) starves difficult multi-hop queries of critical bridging evidence. In this work, we propose the **Evidence-State Marginal Retrieval Controller**, a lightweight post-retrieval framework that observes an initial batch of five documents ($D_5$), extracts seven statistical evidence-state features from the embedding space, and dynamically decides whether to early-exit (**STOP**) or fetch an additional batch of five documents (**MORE**). We supervise the controller using a marginal-gain training oracle ($\Delta F1 > \delta$) strictly computed offline. Evaluated across 1,000 multi-hop HotpotQA queries under a frozen train/validation/test split, our controller achieves an **F1 score of 0.5560** on unseen test data—outperforming both Fixed RAG-5 (0.5443) and Fixed RAG-10 (0.5417)—while achieving a **95.3% early-exit STOP rate**, reducing input token consumption by **48%**, and cutting generation latency by **29%**.

---

## 1. Introduction
Retrieval-Augmented Generation (RAG) significantly reduces hallucinations in Large Language Models (LLMs) by grounding generation on retrieved evidence. However, real-world deployment faces a fundamental efficiency–quality dilemma:
1. **The Cost and Distractor Risk of Static Depth:** Setting a high fixed depth ($k=10$) doubles computational latency and input token consumption. Furthermore, redundant or irrelevant passages act as *semantic distractors*, overwhelming the model's context window and triggering hallucinations.
2. **Insufficiency of Query-Only Routing:** Prior methods such as Adaptive-RAG route queries based purely on pre-retrieval question complexity. However, query text alone cannot reveal whether the retriever actually located the required evidence or whether the retrieved passages are mutually redundant.

To resolve this, we present an evidence-state stopping controller that inspects what retrieval *actually returned* at $k=5$. By evaluating ranking margins, similarity distribution, and pairwise inter-document redundancy, our controller predicts whether a second batch of documents will provide material answer-quality improvement ($\Delta F1 > \delta$).

---

## 2. Problem Formulation & Decision Objective

Given a question $q$ and corpus $\mathcal{C}$, the system retrieves an initial batch $D_5 = \{d_1, \dots, d_5\}$ with dense similarity scores $s_1 \ge s_2 \ge \dots \ge s_5$.

### 2.1 Evidence-State Features ($X_5$)
The controller extracts a 7-dimensional feature vector $X_5 = \Phi(q, D_5)$:
1. **Top-1 Score:** $s_1 = \max_{i} s_i$ (absolute confidence of top document).
2. **Mean Score:** $\bar{s} = \frac{1}{5} \sum_{i=1}^5 s_i$ (overall retrieval alignment).
3. **Score Dispersion:** $\sigma_s = \sqrt{\frac{1}{5}\sum_{i=1}^5 (s_i - \bar{s})^2}$.
4. **Top-1 to Top-2 Margin:** $\Delta s_{12} = s_1 - s_2$ (ranking decisiveness).
5. **Score Range:** $s_1 - s_5$ (evidence quality drop-off).
6. **Max Similarity:** $\max_{i} \cos(\mathbf{e}_q, \mathbf{e}_{d_i})$.
7. **Mean Pairwise Redundancy:** $\frac{1}{10} \sum_{1 \le i < j \le 5} \cos(\mathbf{e}_{d_i}, \mathbf{e}_{d_j})$ (information diversity).

*Zero-Leakage Guarantee:* No gold answers, gold document IDs, or future generation metrics are accessible during feature computation.

### 2.2 Offline Marginal-Gain Oracle
At training time, the system observes both $D_5$ and $D_{10} = D_5 \cup D_{6:10}$, generating candidate answers $a_5$ and $a_{10}$. The ground-truth training label $y \in \{\text{STOP}, \text{MORE}\}$ is:
$$y = \begin{cases} \text{MORE} & \text{if } F1(a_{10}, y^*) - F1(a_5, y^*) > \delta \\ \text{STOP} & \text{otherwise} \end{cases}$$
where $\delta = 0.05$ defines the material improvement threshold.

### 2.3 Inference Flow
At test time, the controller predicts probability $P(\text{MORE} \mid X_5)$. If $P(\text{MORE} \mid X_5) \ge \tau$, the system retrieves $D_{6:10}$ and generates from $D_{10}$; otherwise, it early-exits and generates from $D_5$.

---

## 3. Experimental Setup
- **Dataset:** HotpotQA Multi-hop QA (Dev Distractor set).
- **Splits:** 700 Train, 150 Validation, 150 Frozen Test (stratified by oracle label, seed = 42).
- **Retriever:** Dense inner-product search over normalized embeddings (`all-MiniLM-L6-v2` / `qwen3-embedding:4b`) with FAISS IndexFlatIP.
- **Generator:** Instruction-tuned LLM (`qwen2.5:7b`) with deterministic zero-shot template.
- **Controller:** Gradient Boosting Decision Tree (GBDT, 100 estimators, max depth 3) calibrated with validation threshold $\tau = 0.3$.

---

## 4. Main Results

### 4.1 Evaluation on Frozen Test Split ($N=150$ Unseen Queries)

| Method | Baseline ID | Exact Match | F1 Score | Avg Docs | Total Tokens | Latency | STOP Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Direct LLM | B0 | 12.00% | 0.2187 | 0.00 | 98.2 | 0.41s | 100.0% |
| Fixed RAG-3 | B1 | 36.67% | 0.5394 | 3.00 | 480.7 | 1.37s | — |
| Fixed RAG-5 | B2 | 34.67% | 0.5443 | 5.00 | 761.8 | 1.67s | — |
| Fixed RAG-10 | B3 | 36.00% | 0.5417 | 10.00 | 1523.7 | 2.39s | — |
| Simple Score Rule | B4 | 36.00% | 0.5387 | 7.10 | 1076.6 | 1.93s | 58.0% |
| Adaptive-RAG (Jeong et al.) | B5 | 34.67% | 0.5249 | 6.60 | 1003.6 | 1.92s | 68.0% |
| **Proposed Adaptive-RAG** | **B6** | **36.00%** | **0.5560** | **5.23** | **796.1** | **1.70s** | **95.3%** |
| *Oracle Upper Bound* | *E5* | *40.67%* | *0.6140* | *5.73* | *865.8* | *1.78s* | *85.3%* |

### 4.2 Full Dataset Evaluation ($N=1,000$ Queries)

| Method | Baseline ID | Exact Match | F1 Score | Avg Docs | Total Tokens | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Direct LLM | B0 | 12.40% | 0.2325 | 0.00 | 98.8 | 0.45s |
| Fixed RAG-3 | B1 | 39.00% | 0.5483 | 3.00 | 476.1 | 1.34s |
| Fixed RAG-5 | B2 | 39.80% | 0.5681 | 5.00 | 758.0 | 1.62s |
| Fixed RAG-10 | B3 | 40.50% | 0.5739 | 9.98 | 1544.1 | 2.43s |
| Simple Score Rule | B4 | 39.50% | 0.5666 | 7.02 | 1072.6 | 1.91s |
| Adaptive-RAG (Jeong et al.) | B5 | 40.10% | 0.5683 | 6.91 | 1066.5 | 1.95s |
| **Proposed Adaptive-RAG** | **B6** | **42.50%** | **0.5958** | **5.28** | **800.8** | **1.65s** |

### 4.3 Key Empirical Findings
1. **The Context Bloat Trap:** On unseen test data, increasing retrieval from $k=5$ to $k=10$ actually degrades F1 from 0.5443 down to 0.5417 while doubling token costs (761.8 $\to$ 1523.7).
2. **Quality Superiority:** The proposed controller achieves **0.5560 F1**, outperforming Fixed RAG-10, Fixed RAG-5, and Adaptive-RAG. By actively stopping when evidence is already sufficient, it shields the LLM from distractor passages.
3. **Massive Efficiency Gains:** B6 achieves a 95.3% STOP rate on the test split, consuming only 796 tokens per query (48% reduction vs $k=10$) with average latency of 1.70s vs 2.39s.

---

## 5. Ablation Studies

| Ablation ID | Description | Test F1 | Test EM | Tokens | STOP Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **A0** | **Full V1 Features (Proposed GBDT)** | **0.5560** | **36.00%** | **796.1** | **95.3%** |
| A1 | Drop Redundancy (No `mean_redundancy`) | 0.5560 | 36.00% | 797.5 | 94.7% |
| A2 | Drop Score/Margin (Only similarity & redundancy) | 0.5475 | 35.33% | 805.2 | 94.0% |
| A3 | Score Features Only (Drop similarity & redundancy) | 0.5560 | 36.00% | 800.4 | 94.7% |
| A4.1 | Controller = Logistic Regression | 0.5417 | 36.00% | 1514.8 | 1.3% |
| A4.2 | Controller = Random Forest | 0.5418 | 36.00% | 1494.0 | 3.3% |
| A5.1 | $\delta = 0.00$ Sensitivity | 0.5494 | 35.33% | 804.3 | 94.7% |
| A5.2 | $\delta = 0.02$ Sensitivity | 0.5568 | 36.00% | 797.2 | 95.3% |
| A5.3 | $\delta = 0.10$ Sensitivity | 0.5560 | 36.00% | 802.1 | 94.7% |
| A6.1 | Threshold $\tau = 0.1$ | 0.5362 | 36.00% | 1167.6 | 44.0% |
| A6.2 | Threshold $\tau = 0.2$ | 0.5640 | 36.67% | 836.9 | 89.3% |
| A6.3 | Threshold $\tau = 0.4$ | 0.5535 | 35.33% | 779.4 | 97.3% |
| A6.4 | Threshold $\tau = 0.5$ | 0.5491 | 35.33% | 775.2 | 98.0% |
| A6.5 | Threshold $\tau = 0.9$ | 0.5443 | 34.67% | 761.8 | 100.0% |
| **E5** | **Oracle Upper Bound (Perfect Hindsight)** | **0.6140** | **40.67%** | **865.8** | **85.3%** |

### Insights from Ablations:
- **Importance of Score Distribution (A2):** Dropping retrieval score features (`top1_score`, `margin_top1_top2`, `score_range`, `score_std`) causes F1 to drop by 0.0085, confirming that retrieval score shape is the primary driver of evidence sufficiency.
- **Controller Model Capacity (A4):** Linear logistic regression fails under class imbalance without nonlinear feature interactions, resulting in over-retrieval ($k=10$ on 98.7% of queries) and degraded accuracy.
- **Threshold Calibration (A6):** Setting $\tau \in [0.2, 0.3]$ provides the optimal operating point on the Pareto frontier.

---

## 6. Error Analysis & Failure Taxonomy
Inspection of misclassified instances reveals three distinct failure regimes:
1. **Metric Artifact (False MORE):** As documented in `docs/DISTRACTOR_EXAMPLES.md`, rigid Token-F1 occasionally rewards partial token overlap on hallucinated answers, creating noisy training labels.
2. **Missing Bridging Entity (False STOP):** Complex multi-hop chains where $D_5$ retrieved only the first hop with misleadingly high cosine similarity, causing the controller to stop before the bridging document was fetched.
3. **Semantic Distractors:** When expanding to $k=10$, newly retrieved documents containing entity name collisions tricked the LLM into answering with the distractor entity.

---

## 7. Limitations & Future Work
- **Discrete 5+5 Slicing:** Our V1 architecture makes decisions at the $k=5$ boundary. Future work will investigate continuous token-level stopping ($k \in \{1, 2, 3, \dots\}$).
- **Retriever Dependence:** Controller accuracy relies on dense cosine similarity calibration. Cross-encoder rerankers may provide even sharper margin signals.

---

## 8. Conclusion
We demonstrated that post-retrieval evidence-state features enable a lightweight controller to predict whether expanded retrieval will improve multi-hop answer quality. On HotpotQA, our controller outperforms both Fixed RAG-5 and Fixed RAG-10 while cutting token consumption in half.
