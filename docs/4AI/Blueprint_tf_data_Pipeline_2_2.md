### **Blueprint for `tf.data` Pipeline Implementation (2/2): Pipeline Orchestration**

**Objective:** Replace the `pandas.apply`-based iteration in `VNLPipeline` with a `tf.data`-powered batch processing loop that utilizes the new `.predict_batch()` model methods.

---

#### **Phase 2: Re-architect `VNLPipeline` (`pipeline_colab.py`)**

This phase focuses on creating a new, high-performance data processing workflow. The old row-by-row method can be kept for backward compatibility or removed entirely.

**2.1. New High-Performance `process_dataframe` Method**

*   **File to Modify:** `vnlp_colab/pipeline_colab.py`
*   **New Method Signature:** `process_dataframe(self, df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame`
*   **Implementation Plan:**
    1.  **Input:** The method will take a preprocessed DataFrame (which must contain the `tokens`, `sentence`, and `no_accents` columns) and a `batch_size`.
    2.  **Generator Function:** A private helper function, `_dataframe_generator`, will be created. It will iterate through the input DataFrame and `yield` the required data for each row as raw TensorFlow constants (`tf.constant`). This avoids loading the entire dataset into memory as tensors.
    3.  **`tf.data.Dataset` Creation:**
        *   The main method will instantiate a `tf.data.Dataset` using `tf.data.Dataset.from_generator()`, pointing it to the new generator function.
        *   Crucially, `output_signature` will be defined to inform TensorFlow of the shape and type of the data being yielded (e.g., `tf.TensorSpec(shape=(None,), dtype=tf.string)` for the variable-length token lists).
    4.  **Batching and Prefetching:**
        *   The `.padded_batch(batch_size, padded_shapes=...)` method will be applied to the dataset. This is the core of the optimization. It will group sentences into batches and pad the `tokens` dimension to the length of the longest sentence *in that batch*, creating a rectangular tensor that the GPU can process efficiently.
        *   The `.prefetch(tf.data.AUTOTUNE)` method will be chained at the end. This is a critical performance optimization that allows the CPU to prepare the next batch of data while the GPU is working on the current one, minimizing GPU idle time.
    5.  **Batch Iteration Loop:**
        *   The method will iterate over the prepared `dataset` object (`for batch in dataset:`). A `tqdm` progress bar will be used to track batch processing.
        *   Inside the loop, for each `batch`, the tensors will be converted back to standard Python lists of strings (`.numpy().tolist()`, `.decode()`).
        *   Each model's new `.predict_batch()` method will be called sequentially on the prepared batch data.
        *   The results from each batch will be appended to master lists (e.g., `all_pos_results.extend(batch_pos_results)`).
    6.  **Final Integration:** After the loop finishes, the collected master lists of results will be assigned as new columns to the original input DataFrame. Post-processing steps (like converting lists of tuples to simple lists of tags) will be performed vectorized on the entire column for efficiency.
    7.  **Return:** The final, fully annotated DataFrame is returned.

**2.2. Update the Main `run` Method**

*   **File to Modify:** `vnlp_colab/pipeline_colab.py`
*   **Implementation Plan:**
    1.  The main `run` method's signature will be updated to accept a `batch_size` parameter with a sensible default (e.g., `32`).
    2.  It will be modified to call the new `process_dataframe` method instead of the old `run_analysis` (which used `.apply`).
    3.  It will include timing logic to measure the total execution time of `process_dataframe` and calculate the average time per row, providing a clear performance metric to the user.

**2.3. Deprecation of `run_analysis`**

*   The old `run_analysis` method, which contains the `progress_apply` loops, will be removed or marked as deprecated. The new `process_dataframe` method is its direct, superior replacement.

This new architecture fundamentally changes the data flow from a "pull" model (Pandas pulling one row at a time) to a "push" model (a `tf.data` pipeline pushing optimized batches to the GPU). This is the standard and correct way to build high-throughput inference systems in the TensorFlow ecosystem and will yield the desired performance improvements.