### **Blueprint for Advanced Inference with ONNX (2/2): Runtime Implementation**

**Objective:** Integrate the loaded `onnxruntime.InferenceSession` into the batch processing workflow, replacing the Keras/TensorFlow prediction calls with faster ONNX-based execution.

---

#### **Phase 4: Implementing ONNX Prediction in Batch Methods**

This is where the performance gain is realized. We will modify the `.predict_batch()` methods (developed for the `tf.data` pipeline) to use the `onnxruntime` session.

**4.1. `onnxruntime` Prediction Logic**

*   **Core Method:** `InferenceSession.run()`
*   **Signature:** `run(output_names, input_feed)`
    *   `output_names`: A list of output node names to compute. We'll get this from `session.get_outputs()`.
    *   `input_feed`: A **dictionary** mapping input node names (strings) to their corresponding NumPy array data.
*   **Key Change:** Instead of calling `self.compiled_predict_step(*list_of_tensors)`, we will now construct a dictionary and call `self.inference_session.run(...)`.

**4.2. Refactoring the Autoregressive `.predict_batch()` Methods**

*   **Files to Modify:** `pos_colab.py`, `ner_colab.py`, `dep_colab.py`.
*   **Implementation Plan:** The overall structure of the autoregressive loop (`for t in range(max_sentence_length):`) remains the same. The change happens *inside* the loop.
    1.  **Data Preparation:** The code that prepares the input tensors for the `t`-th token of each sentence in the batch remains unchanged. It will produce a list of NumPy arrays.
    2.  **Input Feed Creation (New Step):** Before calling the model, create the `input_feed` dictionary.
        ```python
        # Inside the loop of predict_batch in pos_colab.py
        
        # list_of_numpy_arrays = [word_batch, left_ctx_batch, right_ctx_batch, lc_pos_history_batch]
        # self.input_names = ['word_input', 'left_input', 'right_input', 'lc_pos_input']

        input_feed = dict(zip(self.input_names, list_of_numpy_arrays))
        ```
    3.  **Execution (Modified Step):** Replace the TensorFlow call with the `onnxruntime` call.
        ```python
        # OLD way:
        # predictions = self.compiled_predict_step(*list_of_tensors).numpy()

        # NEW way:
        predictions = self.inference_session.run([self.output_name], input_feed)[0]
        ```
        *Note: `session.run()` returns a list of outputs, so we take the first element `[0]`.*
    4.  **Post-processing:** The rest of the logic (unstacking predictions, updating state for the next step) remains the same.

---

#### **Phase 5: Workflow Summary & Handoff**

This ONNX migration represents the peak of the performance optimization effort.

**Complete End-to-End Workflow after ONNX:**

1.  **Offline (Developer Task):**
    *   Load the final Keras models.
    *   Use `tf2onnx` to convert each one to an `.onnx` file.
    *   Upload these `.onnx` files to the S3 bucket or other remote storage.

2.  **Runtime (User Execution in `vnlp-colab`):**
    *   `VNLPipeline` is initialized.
    *   Model classes (e.g., `SPUContextPoS`) are initialized.
        *   They download the `.onnx` file (instead of `.weights`).
        *   They create an `onnxruntime.InferenceSession` on the GPU (instead of a Keras model).
    *   The `VNLPipeline.process_dataframe` method is called.
        *   It creates a `tf.data` pipeline to efficiently prepare batches of data.
        *   It iterates through the batches.
        *   For each batch, it calls the model's `.predict_batch()` method.
        *   The `.predict_batch()` method runs its autoregressive loop, and at each step, it calls `self.inference_session.run()` to execute the highly optimized ONNX model on the batched data.

This architecture combines the best of both worlds: TensorFlow's powerful and flexible `tf.data` API for data loading and preprocessing, and `onnxruntime`'s specialized, high-speed engine for the core computational task of inference.