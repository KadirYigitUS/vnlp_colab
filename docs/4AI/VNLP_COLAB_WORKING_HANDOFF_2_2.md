### **`vnlp_colab` Handoff Document (2/2): Technical Specifications & Architecture**

**Project Title:** `vnlp-colab` High-Performance Refactoring
**Version:** 2.1-planning
**Date:** 2025-11-14

---

#### **5. Current System Architecture**

The `vnlp-colab` package is designed with a modular, three-layer architecture:

1.  **Utility & Resource Layer (`vnlp_colab/utils_colab.py`)**
    *   **Purpose:** The foundational layer providing shared services.
    *   **Components:**
        *   `setup_logging()`: Configures structured logging.
        *   `get_vnlp_cache_dir()`: Manages the external cache for large downloaded files (e.g., model weights).
        *   `get_resource_path()`: **Crucial function** that uses `importlib.resources` to reliably locate data files (`.json`, `.txt`, `.model`) bundled within the installed package.
        *   `download_resource()`: Handles downloading of large assets with progress bars.
        *   `load_keras_tokenizer()`: A helper for loading Keras tokenizer configurations.
        *   `create_rnn_stacks()` / `process_word_context()`: Keras 3 helpers for building recurring model components.

2.  **Model Layer (e.g., `vnlp_colab/pos/`, `vnlp_colab/ner/`)**
    *   **Purpose:** Encapsulates the logic for each specific NLP task.
    *   **Standard Structure (per task):**
        *   `*_colab.py`: The main interface file. Contains a user-facing factory class (e.g., `PoSTagger`) that implements the singleton pattern to manage model instances. This file also contains the high-level model implementation classes (e.g., `SPUContextPoS`).
        *   `*_utils_colab.py`: Contains the Keras 3 Functional API code (`create_*_model` functions) that defines the neural network architecture.
        *   `resources/`: A directory (configured as a Python package with `__init__.py`) containing any necessary data files like tokenizers.

3.  **Orchestration Layer (`vnlp_colab/pipeline_colab.py`)**
    *   **Purpose:** Provides the main user-facing API, the `VNLPipeline` class.
    *   **Current Workflow:**
        1.  **Initialization:** The user specifies which models to load. The `__init__` method resolves dependencies (e.g., `TreeStackDP` requires `TreeStackPoS`) and instantiates all required models via their singleton factories.
        2.  **Preprocessing (`run_preprocessing`)**: Takes a DataFrame and uses `pandas.apply` to clean text and tokenize sentences, creating the `tokens` column.
        3.  **Analysis (`run_analysis` / `process_dataframe`):** This is the **current bottleneck**. It iterates through the DataFrame row-by-row and calls the `.predict()` method of each model for each sentence.
        4.  **Execution (`run`):** A wrapper method that calls the preprocessing and analysis steps in sequence.

---

#### **6. Technical Specifications & Constraints**

*   **Runtime Environment:** Google Colab
    *   **Python:** >= 3.10
    *   **TensorFlow:** >= 2.15
    *   **Keras:** >= 3.0 (as part of TensorFlow)
    *   **Hardware:** Optimized for a T4 GPU.
*   **Key Dependencies:** `tensorflow`, `keras`, `pandas`, `numpy`, `sentencepiece`.
*   **Core Design Principles:**
    *   **Singleton Pattern:** Heavy models are loaded only once per session.
    *   **"Tokenize Once":** The orchestrator is responsible for tokenization; models receive pre-tokenized lists.
    *   **Modularity:** Each component is self-contained and testable.
    *   **Keras 3 Functional API:** All models are defined using the modern, explicit API for clarity and forward compatibility.
    *   **GPU Utilization:** All deep learning components are designed to run on the GPU.

---

#### **7. Handoff Summary for Next Session**

The immediate and sole focus of the next development session is to **eliminate the `pandas.apply` bottleneck**. This involves re-architecting the `VNLPipeline` and the `predict` methods of the core models to support **batched inference**. The goal is to process dozens or hundreds of sentences simultaneously in parallel on the GPU, rather than one at a time. This will be achieved by migrating the core processing loop to the `tf.data` API.