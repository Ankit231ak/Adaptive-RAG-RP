# Baseline Inspection: Distractor / Context Bloat Examples

**Date Added:** 11 September 2026
**Finding:** Evidence of `k=10` causing answer degradation compared to `k=5`.

### Example 1: Verbosity / Format Loss
**Question ID:** `5ab97d0a5542996be202051e`
**Question:** "Who was hung for assisting the attempted surrender of a defector from the American Continental Army to the British Army?"
**Gold Answer:** "John André"

**B2_Fixed_RAG_5 (k=5)**
* **Predicted:** "John André"
* **Input Tokens:** 770
* **F1 Score:** 1.0 (Exact Match)

**B3_Fixed_RAG_10 (k=10)**
* **Predicted:** "John André was hung for assisting the attempted surrender of a defector from the American Continental Army to the British Army."
* **Input Tokens:** 1694
* **F1 Score:** 0.2105 (Failed Exact Match)

**Analysis for Paper:**
Retrieving 10 documents increased the input context by 120%. This massive increase in context noise distracted the LLM, causing it to drop its implicit strict-formatting instructions and become conversational. As a result, the HotpotQA token-level F1 score plummeted from 1.0 down to 0.21. 

**Oracle Impact:** 
`∆F1 = 0.2105 - 1.0 = -0.7895`. The Oracle will correctly assign this question a **STOP** label to prevent this degradation.

---

### Example 2: Progressive Context Degradation
**Question ID:** `5ac19f405542991316484b5b`
**Question:** "Pandikona and Berger Blanc Suisse are both what kinds of animals?"
**Gold Answer:** "dogs"

**B1_Fixed_RAG_3 (k=3)**
* **Predicted:** "dogs"
* **Input Tokens:** 347
* **F1 Score:** 1.0 (Perfect)

**B2_Fixed_RAG_5 (k=5)**
* **Predicted:** "Pandikona and Berger Blanc Suisse are both dogs."
* **Input Tokens:** 665
* **F1 Score:** 0.2222 (Verbosity penalty)

**B3_Fixed_RAG_10 (k=10)**
* **Predicted:** "Pandikona and Berger Blanc Suisse are both dog breeds."
* **Input Tokens:** 1431
* **F1 Score:** 0.0 (Failed due to "dogs" vs "dog breeds" token mismatch)

**Analysis for Paper:**
This example demonstrates a perfect "monotonic degradation curve"—the more documents we feed the LLM, the worse the output gets. At `k=3` (347 tokens), the model obeys the implicit instruction for a concise entity answer ("dogs"). As context bloats to `k=5` and `k=10` (1431 tokens), the model loses its instruction-following capability, becoming conversational and slightly altering the entity to "dog breeds," which results in a fatal 0.0 F1 score. 

**Oracle & Future Work Impact:**
Our V1 Oracle evaluates `k=5` vs `k=10`. Here, `∆F1 = 0.0 - 0.2222 = -0.2222`, so our Oracle correctly predicts **STOP** to prevent the fatal 0.0 drop. However, this case study explicitly highlights a limitation for the "Future Work" section of the paper: our V1 system uses fixed batches of 5, but optimal stopping for some queries occurs as early as `k=3`. A future extension of this controller could evaluate marginal gain continuously at `k=1, 2, 3...`.

---

### Example 3: Positive Retrieval Gain (The "MORE" Case)
**Question ID:** `5adcdf455542994ed6169c11`
**Question:** "What where both Hawker Hurricane and No. 1455 Flight apart of?"
**Gold Answer:** "Royal Air Force"

**B2_Fixed_RAG_5 (k=5)**
* **Predicted:** "Royal Air Force (RAF)"
* **Input Tokens:** 656
* **F1 Score:** 0.8571 (Penalized for acronym addition)

**B3_Fixed_RAG_10 (k=10)**
* **Predicted:** "Royal Air Force"
* **Input Tokens:** 1140
* **F1 Score:** 1.0 (Exact Match)

**Analysis for Paper:**
While the previous examples demonstrated the danger of context bloat, this example proves the necessity of an *adaptive* system. Here, `k=5` yielded an imperfect score because the LLM hallucinated the acronym "(RAF)". When provided with 5 additional documents (`k=10`), the expanded context steered the generator back to the exact strict entity without the acronym, achieving a perfect 1.0 F1 score.

**Oracle Impact:**
`∆F1 = 1.0 - 0.8571 = +0.1429`. Because the marginal gain is highly positive, the Oracle will correctly label this question as **MORE**, successfully training the ML controller to identify states where expanded retrieval leads to a more precise generation.

---

### Example 4: The Metric Artifact (Crucial Limitation)
**Question ID:** `5ab2659e554299340b5254b2`
**Question:** "From March 631 to April 631, Farrukhzad Khosrau V was the king of an empire that succeeded which empire?"
**Gold Answer:** "the Parthian Empire"

**B1_Fixed_RAG_3 (k=3)**
* **Predicted:** "Parthian Empire"
* **F1 Score:** 1.0 (Exact Match)

**B2_Fixed_RAG_5 (k=5)**
* **Predicted:** "From March 631 to April 631, Farrukhzad Khosrau V was the king of the Sasanian Empire, which succeeded the Parthian Empire."
* **F1 Score:** 0.2 (Severely penalized for verbosity, but **factually correct**)

**B3_Fixed_RAG_10 (k=10)**
* **Predicted:** "Sasanian Empire"
* **F1 Score:** 0.5 (Mathematically higher, but **factually wrong**)

**Analysis for Paper:**
This is a critical textbook example of an "Evaluation Metric Artifact," which is explicitly listed in our Risk Register (Section 19), and perfectly illustrates the danger of fixed-k context bloat. 
* At **`k=3`**, the model operates with minimal noise, correctly and concisely generating "Parthian Empire" for a perfect **1.0 F1 score**.
* At **`k=5`**, the added context distracts the LLM into verbosity. It generates a full, factually correct sentence, but the rigid Token-F1 metric brutally penalizes it down to **0.2 F1**.
* At **`k=10`**, the massive context bloat fatally confuses the LLM. It misreads the multi-hop question and hallucinates the wrong empire ("Sasanian Empire"). However, because this wrong answer is short and shares the word "Empire" with the gold answer, the broken metric rewards it with a **0.5 F1 score**!

**Oracle Impact:**
Because `∆F1 = 0.5 - 0.2 = +0.3`, the Oracle will blindly label this as **MORE**, effectively training the controller to fetch 10 documents and hallucinate! This proves two things for the final paper: (1) relying purely on Token-F1 for training labels has limitations that require an "LLM-as-a-judge" semantic evaluator, and (2) optimal routing requires evaluating `k=3` (where the answer was perfect) rather than only deciding between `k=5` and `k=10`.

---

### Example 5: Multi-hop Retrieval Threshold (The Perfect "STOP" at 5)
**Question ID:** `5ae12a4b55429920d52342df`
**Question:** "Baraki Barak District is situated in the western part of a province whose capital is what?"
**Gold Answer:** "Puli Alam"

**B1_Fixed_RAG_3 (k=3)**
* **Predicted:** "Kabul"
* **Recall at Retrieved:** 0.5 (Missing critical evidence)
* **F1 Score:** 0.0 (Hallucinated incorrect city)

**B2_Fixed_RAG_5 (k=5)**
* **Predicted:** "Puli Alam"
* **Recall at Retrieved:** 1.0 (Found the missing link)
* **F1 Score:** 1.0 (Exact Match)

**B3_Fixed_RAG_10 (k=10)**
* **Predicted:** "Logar Province, whose capital is Puli Alam."
* **F1 Score:** 0.4444 (Verbosity penalty)

**Analysis for Paper:**
This is the ultimate justification for a dynamically stopping controller. At `k=3`, the retriever failed to fetch both required documents for this multi-hop query (evidenced by the 0.5 retrieval recall), forcing the generator to hallucinate "Kabul". By expanding to `k=5`, the missing document was fetched, allowing a perfect 1.0 F1 score. However, expanding further to `k=10` introduced context bloat, causing the LLM to become conversational and tanking the F1 score to 0.44.

**Oracle Impact:**
`∆F1 = 0.4444 - 1.0 = -0.5556`. The Oracle correctly assigns **STOP**. This example proves that `k=3` is not universally optimal, and a controller that reliably stops at `k=5` when the evidence is sufficient is vastly superior to a static `k=3` or `k=10` rule.

---

### Example 6: Deep Multi-Hop / Refusal to Hallucinate (The Perfect "MORE" Case)
**Question ID:** `5ae22eae554299495565da30`
**Question:** "What was the 2010 population of the town where Black Crescent Mountain was located?"
**Gold Answer:** "310"

**B1_Fixed_RAG_3 (k=3)**
* **Predicted:** "The passage does not provide the population of the town where Black Crescent Mountain was located..."
* **Recall at Retrieved:** 0.5 (Missing population document)
* **F1 Score:** 0.0 

**B2_Fixed_RAG_5 (k=5)**
* **Predicted:** "The context does not provide the 2010 population..."
* **Recall at Retrieved:** 0.5 (Still missing document)
* **F1 Score:** 0.0 

**B3_Fixed_RAG_10 (k=10)**
* **Predicted:** "310"
* **Recall at Retrieved:** 1.0 (Successfully fetched missing document)
* **F1 Score:** 1.0 (Exact Match)

**Analysis for Paper:**
This is the textbook example of a deep multi-hop query where initial retrieval fails. At `k=3` and `k=5`, the retriever successfully found the first hop (that the mountain is in Randolph), but failed to retrieve the second hop (Randolph's population). Fortunately, Qwen2.5 correctly refused to hallucinate and admitted the context was missing. Only by extending the retrieval budget to `k=10` was the second hop found, allowing a perfect 1.0 score. 

**Oracle Impact:**
`∆F1 = 1.0 - 0.0 = +1.0`. The Oracle strongly labels this as **MORE**. This example proves that a strict `k=5` cutoff is dangerously low for complex multi-hop queries, completely validating the ML controller's ability to trigger a `MORE` action when it detects poor initial evidence.

---

### Example 7: The Efficiency Baseline (Cost/Latency Savings)
**Question ID:** `5ab6aa4c55429953192ad359`
**Question:** "In the NASA mission where Moon trees were taken into space, what was the nickname of the Command Module?"
**Gold Answer:** "\"Kitty Hawk\""

**B1_Fixed_RAG_3 (k=3)**
* **Predicted:** "Apollo 14"
* **F1 Score:** 0.0 (Hallucinated mission name instead of module nickname due to missing doc)

**B2_Fixed_RAG_5 (k=5)**
* **Predicted:** "Kitty Hawk"
* **F1 Score:** 1.0 (Exact Match)
* **Tokens:** 888
* **Latency:** 4.45s

**B3_Fixed_RAG_10 (k=10)**
* **Predicted:** "Kitty Hawk"
* **F1 Score:** 1.0 (Exact Match)
* **Tokens:** 1616
* **Latency:** 7.33s

**Analysis for Paper:**
While other examples highlight F1 score volatility, this example perfectly highlights the **computational cost constraint**. Both `k=5` and `k=10` produce perfect 1.0 scores. However, `k=10` consumes nearly double the input tokens (1616 vs 888) and takes 65% longer to generate the answer (7.33s vs 4.45s). 

**Oracle Impact:**
`∆F1 = 1.0 - 1.0 = 0.0`. The Oracle recognizes that there is zero marginal gain to retrieving more documents and strictly labels this as **STOP**. This is the primary driver of the Adaptive RAG efficiency curve—by correctly identifying when `D5` is sufficient, the ML controller cuts token cost and compute latency nearly in half without sacrificing any quality.

---

### Example 8: The Fatal Distractor (Factual Hallucination)
**Question ID:** `5a87c13f5542996e4f30890c`
**Question:** "In what city did the 'Prince of tenors' star in a film based on an opera by Giacomo Puccini?"
**Gold Answer:** "Rome"

**B1_Fixed_RAG_3 (k=3)**
* **Predicted:** "Rome"
* **F1 Score:** 1.0 (Exact Match)

**B2_Fixed_RAG_5 (k=5)**
* **Predicted:** "Rome"
* **F1 Score:** 1.0 (Exact Match)

**B3_Fixed_RAG_10 (k=10)**
* **Predicted:** "New York City"
* **F1 Score:** 0.0 (Fatal Factual Hallucination)

**Analysis for Paper:**
While previous context-bloat examples caused the LLM to become overly verbose (resulting in formatting penalties), this example demonstrates the ultimate danger of over-retrieval: **factual hallucination**. The first 5 documents contained clear evidence for "Rome". However, documents 6-10 likely contained semantic distractors (e.g., mentions of tenors or Puccini in New York). The expanded context overwhelmed the LLM's reasoning capabilities, causing it to confidently output the wrong city ("New York City"). 

**Oracle Impact:**
`∆F1 = 0.0 - 1.0 = -1.0`. The Oracle correctly assigns a strong **STOP** label. This proves that an adaptive controller isn't just a cost-saving measure—it acts as a critical hallucination guardrail by isolating the generator from semantic distractors.

---

### Example 9: The Perfect "STOP" for Multi-hop Optimization
**Question ID:** `5ab90d8955429916710eb0f1`
**Question:** "Jalen Jones plays basketball for an NBA team that plays their home games in what arena?"
**Gold Answer:** "Smoothie King Center"

**B1_Fixed_RAG_3 (k=3)**
* **Predicted:** "St. Elizabeths East Entertainment and Sports Arena"
* **Recall at Retrieved:** 0.5 (Missing second hop)
* **F1 Score:** 0.0 (Factual Hallucination/Out of Date info)

**B2_Fixed_RAG_5 (k=5)**
* **Predicted:** "Smoothie King Center"
* **Recall at Retrieved:** 1.0 (Found missing hop)
* **F1 Score:** 1.0 (Exact Match)
* **Tokens / Latency:** 707 tokens / 3.36s

**B3_Fixed_RAG_10 (k=10)**
* **Predicted:** "Smoothie King Center"
* **F1 Score:** 1.0 (Exact Match)
* **Tokens / Latency:** 1465 tokens / 8.53s

**Analysis for Paper:**
This perfectly isolates the balancing act of Adaptive RAG. At `k=3`, the system lacks the second-hop document linking the NBA team to its home arena, causing the LLM to confidently hallucinate an incorrect arena ("St. Elizabeths"). Expanding to `k=5` resolves the missing evidence, achieving a perfect 1.0 score. Expanding further to `k=10` maintains the perfect score, but costs 107% more tokens and takes 2.5x longer to generate (8.5s vs 3.3s). 

**Oracle Impact:**
`∆F1 = 1.0 - 1.0 = 0.0`. The Oracle correctly labels this as **STOP**. The ML controller will learn that once the necessary multi-hop evidence is acquired in the top-5 (detectable via the evidence-state features), it must aggressively STOP to save compute and prevent accidental distractor hallucinations.

---
