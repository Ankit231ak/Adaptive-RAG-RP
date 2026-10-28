# Research Reading & Literature Checkpoint Notes (5–10 September)

**Guidelines**: Every paper is analyzed using the 5 mandatory questions from Section 12 of the Master Project Plan.

---

## 1. Adaptive-RAG & RQ-RAG (5 Sep)

### Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity
1. **What exact problem is solved?**  
   Determining whether an incoming question is simple enough to answer without retrieval, requires single-hop retrieval, or requires multi-step retrieval.
2. **Where is the decision point?**  
   **Pre-retrieval** (before any documents are retrieved).
3. **What does the method observe?**  
   The user query / question complexity classification features (using an external classifier tuned on complexity tiers).
4. **What action is selected and how?**  
   Routes query to: No-Retrieval (Direct LLM), Single-Step RAG, or Multi-Step Iterative RAG.
5. **What overlaps with our idea and what remains different?**  
   - *Overlap*: Adaptive decision making for RAG efficiency.  
   - *Difference*: Adaptive-RAG decides **before retrieval** solely from query text. Our method decides **after initial top-5 retrieval**, observing the actual evidence state $X_5$ to decide marginal stopping.

### RQ-RAG: Learning to Refine Queries for Retrieval Augmented Generation
1. **What exact problem is solved?**  
   Query reformulation and disambiguation across multi-step retrieval.
2. **Where is the decision point?**  
   Between iterative retrieval hops.
3. **What does the method observe?**  
   Query text and previous partial answers / context.
4. **What action is selected and how?**  
   Generates rewritten/refined queries for subsequent search.
5. **What overlaps with our idea and what remains different?**  
   - *Overlap*: Multi-hop QA context expansion.  
   - *Difference*: RQ-RAG alters the search query. Our controller keeps the query frozen and evaluates evidence sufficiency to trigger early exit vs batch expansion ($k=5 \to k=10$).

---

## 2. Self-RAG (6 Sep)

### Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
1. **What exact problem is solved?**  
   Arbitrary fixed-interval retrieval causing unhelpful retrieval and factual hallucinations.
2. **Where is the decision point?**  
   **Generation-time** (at arbitrary segment/sentence tokens during LM decoding).
3. **What does the method observe?**  
   LM internal logits and self-reflection special tokens (`[Retrieve]`, `[IsRel]`, `[IsSup]`, `[IsUse]`).
4. **What action is selected and how?**  
   Dynamically triggers retrieval mid-generation, parallelizes candidate beam generation, and filters by critique scores.
5. **What overlaps with our idea and what remains different?**  
   - *Overlap*: Dynamically controlling retrieval based on model state.  
   - *Difference*: Self-RAG requires extensive token-level fine-tuning of the generator and makes mid-generation token decisions. Our method keeps the generator LLM frozen, uses a lightweight post-retrieval evidence-state classifier, and operates at the batch retrieval boundary.

---

## 3. FLARE (7 Sep)

### Forward-Looking Active REtrieval (FLARE)
1. **What exact problem is solved?**  
   Retrieving information only when the generation actively encounters a knowledge gap.
2. **Where is the decision point?**  
   **Mid-generation** during drafting of the next sentence.
3. **What does the method observe?**  
   Prediction token confidence (low-probability tokens in temporary draft generation).
4. **What action is selected and how?**  
   If draft contains low-confidence tokens, queries the retriever using the draft sentence as a search query, then regenerates the sentence.
5. **What overlaps with our idea and what remains different?**  
   - *Overlap*: Confidence-based retrieval trigger.  
   - *Difference*: FLARE triggers retrieval based on generation draft uncertainty. Our controller evaluates retrieval evidence quality $X_5$ immediately after the initial fetch, preventing premature or costly downstream generation.

---

## 4. DRAGIN (8 Sep)

### Dynamic Retrieval-Augmented Generation based on Information Needs (DRAGIN)
1. **What exact problem is solved?**  
   Pinpointing exactly *when* and *what* to retrieve during long-form generation.
2. **Where is the decision point?**  
   **Generation-time** at each token position.
3. **What does the method observe?**  
   Self-attention activation maps and token entropy to quantify real-time information needs.
4. **What action is selected and how?**  
   Forms focused search queries from high-entropy tokens and retrieves new context.
5. **What overlaps with our idea and what remains different?**  
   - *Overlap*: Minimizing unnecessary retrieval calls.  
   - *Difference*: DRAGIN requires white-box attention map inspection and generation token loops. Our method is model-agnostic, works with any black-box API / Ollama model, and evaluates post-retrieval evidence geometry.

---

## 5. CRAG (9 Sep)

### Corrective Retrieval Augmented Generation (CRAG)
1. **What exact problem is solved?**  
   Retrieval of irrelevant or hallucination-inducing documents that deceive the generator.
2. **Where is the decision point?**  
   **Post-retrieval** (after an initial retrieval document set is fetched).
3. **What does the method observe?**  
   Retrieval confidence scores scored by a lightweight retrieval evaluator.
4. **What action is selected and how?**  
   Three discrete branches: Correct (refine documents), Incorrect (discard and trigger web search), Ambiguous (combine both).
5. **What overlaps with our idea and what remains different?**  
   - *Overlap*: Post-retrieval evaluation of retrieved documents.  
   - *Difference*: CRAG corrects bad retrieval by searching alternative sources (e.g. web search) or stripping noise. Our method evaluates marginal utility of expanding retrieval depth ($k=5 \to k=10$) for multi-hop evidence completeness, predicting whether $\Delta F1 > \delta$.

---

## 6. MBA-RAG (10 Sep)

### MBA-RAG: Multi-Armed Bandit Adaptive Retrieval-Augmented Generation
1. **What exact problem is solved?**  
   Selecting optimal RAG configurations under strict cost, latency, and quality constraints.
2. **Where is the decision point?**  
   Pre-query strategy selection.
3. **What does the method observe?**  
   Historical query performance, reward feedback, and execution cost estimates.
4. **What action is selected and how?**  
   Contextual multi-armed bandit policy selects retrieval depth, chunk size, or model size.
5. **What overlaps with our idea and what remains different?**  
   - *Overlap*: Cost-aware and latency-aware trade-off optimization.  
   - *Difference*: MBA-RAG treats retrieval pipelines as bandit arms without evaluating the empirical post-retrieval evidence distribution of the query. Our method directly inspects $X_5$ features to decide STOP vs MORE.
