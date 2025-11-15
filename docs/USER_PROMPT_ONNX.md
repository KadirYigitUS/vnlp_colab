### **User Prompt for Next Session: ONNX Integration for Peak Performance**

**ROLE:**
You are an **AI Inference Optimization Engineer** specializing in deploying deep learning models for maximum performance. Your task is to complete the final optimization phase for the `vnlp-colab` package. You will convert the existing TensorFlow/Keras models to the ONNX format and refactor the inference code to use the highly optimized `onnxruntime-gpu` engine.

---

### 🎯 **Primary Objectives (Next Session)**

This session is dedicated to **Phase 2: ONNX Conversion and Integration**.

1.  **Add ONNX Dependencies:**
    *   Update the `pyproject.toml` file to include `tf2onnx` and `onnxruntime-gpu` as new dependencies for the project.

2.  **Develop a Model Conversion Utility:**
    *   Create a new script, `vnlp_colab/convert_to_onnx.py`.
    *   This script must contain a main function that systematically loads each of the core Keras models (`Stemmer`, `SPUContextPoS`, `SPUContextNER`, `SPUContextDP`, etc.).
    *   For each loaded Keras model, define its precise `input_signature` using `tf.TensorSpec`, ensuring the `shape`, `dtype`, and `name` attributes match the model's `create_*_model` function. **This step is critical for a successful conversion.**
    *   Use `tf2onnx.convert.from_keras()` to convert the model to the ONNX format (`opset=13`).
    *   Save each converted model to a file (e.g., `PoS_SPUContext_prod.onnx`). The script should be runnable to generate all necessary `.onnx` files.

3.  **Refactor Model Classes for ONNX Runtime:**
    *   Modify the `__init__` method of each model class (e.g., `SPUContextPoS` in `pos_colab.py`, `StemmerAnalyzer` in `stemmer_colab.py`, etc.).
        *   The method should **no longer** build a Keras model in memory (`create_*_model`).
        *   It will now download the corresponding `.onnx` file from remote storage.
        *   It must initialize an `onnxruntime.InferenceSession`, specifying `CUDAExecutionProvider` to ensure GPU execution.
        *   It must programmatically inspect the session to get the exact input and output node names (e.g., `session.get_inputs()`, `session.get_outputs()`) and store them as instance attributes (e.g., `self.input_names`, `self.output_names`).
    *   Refactor the `.predict_batch()` method in each model class.
        *   The core autoregressive or batching loop structure will remain the same.
        *   The call to the TensorFlow model (`self.compiled_predict_step()`) must be **completely replaced** with a call to `self.inference_session.run()`.
        *   This involves creating an `input_feed` dictionary that maps the stored input names to the prepared NumPy batch data for the current step.

4.  **Validate ONNX Pipeline:**
    *   Ensure the fully refactored ONNX-based pipeline passes the existing `pytest` suite. The final output must remain identical to the TensorFlow implementation, confirming a lossless optimization.

---

### 🚫 **DO NOT**

*   Remove the `tf.data` pipeline from `pipeline_colab.py`. The ONNX runtime will consume the batches produced by this pipeline.
*   Hardcode input/output node names. They must be dynamically retrieved from the `InferenceSession`.
*   Leave any Keras model prediction calls (`model.predict`, `@tf.function`) in the main inference path of the `.predict_batch()` methods.