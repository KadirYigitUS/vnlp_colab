### **User Prompt for Next Session: High-Performance Batch Processing Implementation**

**ROLE:**
You are an **AI software engineer** specializing in high-performance machine learning pipelines. Your task is to refactor the `vnlp-colab` package to dramatically increase its inference speed on large datasets. You will achieve this by replacing the current row-by-row `pandas.apply` processing with a fully batched, parallel pipeline leveraging the `tf.data` API.

---

### 🎯 **Primary Objectives (Next Session)**

1.  **Implement Batch Prediction in Models:**
    *   For each core model class (`StemmerAnalyzer`, `SPUContextPoS`, `SPUContextNER`, `SPUContextDP`, and their `TreeStack` counterparts), create a new method named `.predict_batch()`.
    *   This method must accept a *list* of inputs (e.g., a list of token lists) corresponding to a batch of sentences.
    *   For the **autoregressive models** (PoS, NER, DP), the implementation must process the batch "vertically": create a sub-batch of all 1st tokens, predict in parallel; then all 2nd tokens, predict in parallel, and so on. This is the key to maximizing GPU utilization.
    *   The method must return a list of results, with the order matching the input batch.

2.  **Re-architect the `VNLPipeline`:**
    *   Create a new primary processing method named `process_dataframe(self, df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame`.
    *   Inside this method, implement a `tf.data` pipeline:
        *   Use `tf.data.Dataset.from_generator()` to create a dataset from the input DataFrame in a memory-efficient way.
        *   Apply the `.padded_batch()` transformation to group sentences into batches and pad them into rectangular tensors.
        *   Apply the `.prefetch(tf.data.AUTOTUNE)` transformation to create an efficient producer-consumer pipeline between the CPU and GPU.
    *   Iterate over the prepared `tf.data.Dataset`. In each loop, call the new `.predict_batch()` method of each configured model.
    *   Collect the results from all batches and merge them back into the input DataFrame as new columns.

3.  **Update the Main `run` Method:**
    *   Modify the `VNLPipeline.run()` method to accept a `batch_size` parameter.
    *   Replace the call to the old, slow analysis method with a call to the new, high-performance `process_dataframe` method.
    *   Add timing metrics to log the total processing time and the average time per row to clearly demonstrate the performance improvement.

4.  **Holistic Debugging & Validation:**
    *   Ensure all code is fully type-hinted, documented, and adheres to PEP8 standards.
    *   After implementation, perform a test run on a sample dataset of a few hundred rows to confirm that the batched output is identical to the output from the old row-by-row method, ensuring no logical regressions were introduced.

---

### 🚫 **DO NOT**

*   Change the underlying model architectures (`*_utils_colab.py` files).
*   Implement the ONNX conversion in this session. This plan is focused exclusively on the `tf.data` migration.
*   Leave the old `pandas.apply` logic in the main execution path. It should be fully replaced by the new batched method.