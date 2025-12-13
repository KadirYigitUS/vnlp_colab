Technical_Implementation_Blueprint-VNLP_Colab_Refactor.md
## Technical Implementation Blueprint: VNLP Colab Refactor

This blueprint outlines the plan to refactor the VNLP (Turkish Natural Language Processing) package, focusing on creating a modern, efficient, and user-friendly `utils_colab.py` module. The primary goal is to ensure full compatibility with Google Colab's environment, Keras 3, and the latest TensorFlow builds, while enhancing developer experience and maintainability.

### 1. Core Module: `utils_colab.py`

This will be the central utility file, replacing the legacy `utils.py`.

**1.1. Structure & Components:**

*   **Header:**
    *   Module-level docstring (purpose, dependencies, usage).
    *   `VERSION` constant.
    *   Standard library imports (`logging`, `pathlib`, `typing`, etc.).
    *   Third-party imports (`tensorflow`, `numpy`, `requests`, `tqdm`).
*   **Configuration:**
    *   A `logging.basicConfig` setup for clean, structured output in notebooks.
    *   A `CACHE_DIR` constant pointing to `/content/vnlp_cache` for downloaded assets. This leverages HuggingFace-like caching logic.
*   **Hardware Acceleration:**
    *   A function `get_strategy()` that detects available GPUs or TPUs and returns the appropriate `tf.distribute.Strategy`. This ensures models are automatically placed on available accelerators.
*   **Networking & Caching Utilities:**
    *   `check_and_download(file_url: str, cache_dir: Path) -> Path`: A refactored download function.
        *   It will use `pathlib` for path manipulation.
        *   It will download files to the `CACHE_DIR`.
        *   It will use `requests` with streaming and a `tqdm` progress bar for a better user experience during large model downloads.
        *   It will be idempotent, checking for the file's existence before downloading.
*   **Keras/TensorFlow Utilities:**
    *   `load_keras_tokenizer(tokenizer_path: Path) -> tf.keras.preprocessing.text.Tokenizer`: A modernized version that uses `pathlib` and includes robust error handling.
    *   `create_rnn_stacks(...) -> tf.keras.Model`: This function will be refactored to use the Keras 3 Functional API, returning a reusable `tf.keras.Model` instead of a `Sequential` object for better composability. Type hints and clear docstrings will be added.
*   **Text Processing Utilities:**
    *   `tokenize_single_word(...) -> np.ndarray`: Cleaned up with type hints and explicit documentation on padding/truncating behavior.
    *   `process_word_context(...) -> Tuple[np.ndarray, ...]`: This core function will be optimized for vectorization where possible and heavily documented to explain its role in preparing context windows for the models (POS, DEP, NER).
*   **Main Entry Point (Optional):**
    *   An `if __name__ == "__main__":` block with a simple `main()` function demonstrating the usage of the utility functions, primarily for testing purposes.

### 2. Model Architecture and Refactoring Strategy

The analysis will focus on the models that use TensorFlow/Keras directly: `PoSTagger`, `DependencyParser`, `NamedEntityRecognizer`, and `StemmerAnalyzer`.

*   **Model Loading:** All model classes (`SPUContextPOS`, `SPUContextNER`, etc.) will be refactored to use the new `utils_colab.check_and_download` function. Paths will be constructed using `pathlib`.
*   **Keras 3 Compliance:**
    *   Model creation helpers (e.g., `create_spucontext_ner_model`) will be migrated from the `Sequential` API within a `Sequential` block to the pure **Functional API**. This is a critical change for Keras 3 and makes the architecture explicit and easier to debug. For instance, shared layers like `word_rnn` will be defined once and called on multiple inputs.
    *   The model weight loading logic will be hardened. Instead of `model.set_weights(pickle.load(fp))`, which can be brittle, we will use `model.load_weights()` where possible or ensure the pickled weights match the exact architecture of the newly created Keras 3 model. The provided architecture blueprints will be the source of truth.
*   **Performance Optimization:**
    *   `@tf.function`: The prediction loops in the models (which are autoregressive) will have their core model-forward-pass step wrapped in a `@tf.function`. This will compile the step into a static graph, providing a significant speedup in Colab notebooks.
    *   **Vectorization:** The `process_single_word_input` helpers will be reviewed to ensure they use NumPy vector operations instead of Python loops for creating one-hot vectors and padding context arrays.

### 3. Timeline and Deliverables

1.  **Approve Blueprint:** User approves this plan.
2.  **Generate `utils_colab.py`:**
    *   **Part 1:** Imports, logging, caching, and the `check_and_download` function.
    *   **Part 2:** Keras/TF helpers (`load_keras_tokenizer`, `create_rnn_stacks`).
    *   **Part 3:** Text processing helpers (`tokenize_single_word`, `process_word_context`) and final cleanup.
3.  **Provide Changelog and Usage:** A summary of key changes and a simple, copy-pasteable usage example will be provided upon completion.
4.  **Holistic Debug:** A final review of the generated code against all requirements will be performed.

This blueprint ensures a systematic approach to meet all project objectives, prioritizing compatibility, performance, and modern development practices for the Colab environment. I will now await your approval to proceed with generating the first file.