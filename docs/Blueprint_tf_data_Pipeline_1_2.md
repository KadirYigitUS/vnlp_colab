### **Blueprint for `tf.data` Pipeline Implementation (1/2): Core Logic & Model Refactoring**

**Objective:** Rearchitect the `VNLPipeline` and its constituent models to replace row-by-row iteration (`pandas.apply`) with high-performance, batched inference using the `tf.data` API.

---

#### **Phase 1: Refactor Model Interfaces for Batch Processing**

The current `.predict()` methods are designed for a single input. We will introduce new `.predict_batch()` methods designed to accept and process a list of multiple inputs simultaneously.

**1.1. `StemmerAnalyzer` (`stemmer_colab.py`)**

*   **File to Modify:** `vnlp_colab/stemmer/stemmer_colab.py`
*   **New Method Signature:** `predict_batch(self, batch_of_tokens: List[List[str]]) -> List[List[str]]`
*   **Implementation Plan:**
    1.  The existing `process_stemmer_input` utility is already designed to handle multiple sentences. We will create a new, simplified version of the `predict` logic that processes the entire batch at once.
    2.  The method will take a `batch_of_tokens` (e.g., `[['Bu', 'film', '.'], ['Merhaba', 'dünya', '.']]`).
    3.  It will generate analysis candidates for all tokens in all sentences.
    4.  It will call `process_stemmer_input` to create a single large tensor batch for *all tokens from all sentences combined*.
    5.  It will execute a single `self.compiled_predict_step` call on this large batch.
    6.  The results will be partitioned and re-grouped back into a list of lists, corresponding to the input sentences.
    *   **Rationale:** This is simpler than the other models because the stemmer is not autoregressive. We can process all tokens independently.

**1.2. `SPUContextPoS` / `TreeStackPoS` (`pos_colab.py`)**

*   **File to Modify:** `vnlp_colab/pos/pos_colab.py`
*   **New Method Signature:** `predict_batch(self, batch_of_tokens: List[List[str]]) -> List[List[Tuple[str, str]]]`
*   **Implementation Plan (Autoregressive Batching):** This is the most complex but most important part.
    1.  Determine the `max_sentence_length` within the current batch.
    2.  Initialize state for all sentences in the batch (e.g., `batch_int_preds = [[] for _ in batch_of_tokens]`).
    3.  Loop `for t in range(max_sentence_length)`:
        a.  For each sentence `s` in the batch that is long enough (i.e., `len(tokens) > t`), prepare the input for the `t`-th token. This includes the token itself, its left/right context, and the history of previous predictions for that specific sentence `s`.
        b.  **Stack these inputs** from all sentences into a single, large batch tensor.
        c.  Execute a **single** `self.compiled_predict_step` call on the stacked tensors. The model will process the `t`-th token of all sentences in the batch in parallel.
        d.  Unstack the results and append the prediction for each sentence `s` to its corresponding list in `batch_int_preds`.
    4.  After the loop, convert all integer predictions to labels and zip them with the original tokens for each sentence.
    *   **Rationale:** This "vertical" batching (processing all 1st tokens, then all 2nd tokens, etc.) is the correct way to parallelize autoregressive models. It keeps the GPU saturated at every step of the sequence generation.

**1.3. `SPUContextNER` / `CharNER` (`ner_colab.py`)**

*   **File to Modify:** `vnlp_colab/ner/ner_colab.py`
*   **New Method Signature:** `predict_batch(self, batch_of_sentences: List[str], batch_of_tokens: List[List[str]]) -> List[List[Tuple[str, str]]]`
*   **Implementation Plan:**
    *   **For `SPUContextNER`:** The implementation will be identical to the autoregressive batching logic described for `SPUContextPoS`.
    *   **For `CharNER`:** This model is not autoregressive. Its `predict_batch` method will be simpler:
        1.  Loop through each sentence in the batch.
        2.  Call the existing single-sentence `predict` method.
        3.  Collect and return the results. (Note: `CharNER` is less critical to optimize as it is not the default model, but providing the batch API maintains consistency).

**1.4. `SPUContextDP` / `TreeStackDP` (`dep_colab.py`)**

*   **File to Modify:** `vnlp_colab/dep/dep_colab.py`
*   **New Method Signature:** `predict_batch(self, batch_of_tokens: List[List[str]], pos_result_batch: List[List[Tuple[str, str]]]) -> List[List[Tuple[int, str, int, str]]]`
*   **Implementation Plan:** The implementation will be identical to the autoregressive batching logic described for `SPUContextPoS` and `SPUContextNER`. The state maintained at each step will include both `arcs` and `labels` for each sentence in the batch.