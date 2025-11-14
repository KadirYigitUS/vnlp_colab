### **Blueprint for Advanced Inference with ONNX (1/2): Concept & Conversion**

**Objective:** Achieve the next tier of inference performance by converting the trained Keras models to the ONNX (Open Neural Network Exchange) format and using the highly optimized `onnxruntime` engine for execution.

**Prerequisite:** The `tf.data` pipeline refactoring (Blueprint parts 3 & 4) must be completed first. The ONNX engine will replace the Keras `model()` call, but it still relies on the same batched data input.

---

#### **Phase 1: Understanding ONNX and `onnxruntime`**

*   **What is ONNX?** ONNX is an open standard format for representing machine learning models. It acts as a universal translator, allowing models trained in one framework (like TensorFlow/Keras) to be run in another (like PyTorch) or, more importantly, in a dedicated, high-speed inference engine.
*   **What is `onnxruntime`?** This is Microsoft's cross-platform, high-performance ML inference engine. It is specifically designed to run ONNX models at maximum speed. It can leverage hardware-specific acceleration libraries (like CUDA and TensorRT on NVIDIA GPUs) more aggressively than the general-purpose TensorFlow runtime, often resulting in significant speedups (1.2x to 3x or more) with no loss in accuracy.
*   **Why is this the next step?** While `@tf.function` is fast, `onnxruntime` is specialized *only* for inference. It strips away all training-related components, performs advanced graph optimizations (like layer fusion), and links directly to the most optimized hardware kernels available.

---

<h4>**Phase 2: The Conversion Process (`tf2onnx`)**</h4>

We will use the `tf2onnx` library to convert our saved Keras models into `.onnx` files. This is a one-time, offline process that happens *after* a model is trained or fine-tuned.

**2.1. New Dependency**

*   A new dependency, `tf2onnx`, will be added to the project's `pyproject.toml` or installed directly in the environment. We will also need `onnxruntime-gpu`.
    ```bash
    pip install tf2onnx onnxruntime-gpu
    ```

**2.2. Conversion Script/Logic**

*   **Goal:** Create a script or utility function that can take a Keras model and export it as an `.onnx` file.
*   **Input:** A fully loaded Keras model instance (e.g., from `pos_colab.py`).
*   **Core Command:** The conversion is typically a single function call.
    ```python
    import tf2onnx
    import tensorflow as tf

    # 1. Load the Keras model
    keras_model = ... # e.g., PoSTagger().instance.model

    # 2. Define the input signature. This is CRITICAL.
    # It must match the model's input shapes exactly.
    input_signature = [
        tf.TensorSpec(shape=(None, 8), dtype=tf.int32, name="word_input"),
        tf.TensorSpec(shape=(None, 40, 8), dtype=tf.int32, name="left_context_input"),
        # ... and so on for all inputs
    ]

    # 3. Convert and save the model
    # opset=13 is a good, stable choice for compatibility.
    model_proto, _ = tf2onnx.convert.from_keras(keras_model, input_signature, opset=13)
    with open("pos_model.onnx", "wb") as f:
        f.write(model_proto.SerializeToString())
    ```
*   **Workflow Integration:** This conversion process is an **asset generation step**. We would run this for each of our final, trained models (`PoS`, `NER`, `DP`, etc.) and save the resulting `.onnx` files. These files would then be uploaded to a remote store (like the S3 bucket where the `.weights` files are currently stored) to be downloaded by the `vnlp-colab` package at runtime.

---

#### **Phase 3: Refactor Model Classes for ONNX Inference**

This phase involves modifying the model implementation classes (e.g., `SPUContextPoS`) to use `onnxruntime` instead of TensorFlow for the actual prediction step.

**3.1. New `__init__` Logic**

*   **File to Modify:** All `*_colab.py` model files (e.g., `pos_colab.py`).
*   **Implementation Plan:**
    1.  The `__init__` method will no longer call `create_spucontext_pos_model()` to build a Keras model in memory.
    2.  Instead, it will download the corresponding `.onnx` file (e.g., `PoS_SPUContext_prod.onnx`) from the remote store.
    3.  It will then create an `onnxruntime.InferenceSession`. This session object loads the ONNX model graph and prepares it for execution on the GPU.
        ```python
        import onnxruntime as ort

        onnx_model_path = download_resource("PoS_SPUContext_prod.onnx", "...")
        
        # Specify GPU execution provider
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.inference_session = ort.InferenceSession(str(onnx_model_path), providers=providers)

        # We also need to get the exact input names from the model
        self.input_names = [inp.name for inp in self.inference_session.get_inputs()]
        self.output_name = self.inference_session.get_outputs()[0].name
        ```
    *   The Keras model (`self.model`) and the compiled TensorFlow function (`self.compiled_predict_step`) will be completely removed. The `self.inference_session` is their direct replacement.