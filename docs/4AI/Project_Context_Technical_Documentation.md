## **Part 1: Project & Business Context Documents**

These documents establish the strategic foundation of the `vnlp-colab` project, clarifying its purpose, scope, and specific requirements for all stakeholders.

### **1. Project Charter: `vnlp-colab`**

**1.1. Project Vision & Mission**

*   **Vision:** To democratize advanced Turkish Natural Language Processing by providing a high-performance, accessible, and easy-to-use toolkit optimized for collaborative research and development environments.
*   **Mission:** To refactor the existing VNLP library into a modern, Keras 3-native package (`vnlp-colab`) that runs seamlessly and efficiently on Google Colab. The project will prioritize performance on free-tier hardware (T4 GPUs), enhance developer experience through modern tooling, and maintain functional parity with the original library while improving its architectural design.

**1.2. Problem Statement**

The original VNLP library, while powerful, was built on legacy TensorFlow 1.x/Keras 2.x patterns. This presents several challenges for modern users:
1.  **Compatibility Issues:** It suffers from dependency conflicts and deprecated API calls in current environments like Google Colab, which uses TensorFlow 2.x and Keras 3.
2.  **Performance Bottlenecks:** Its architecture contains unoptimized Python loops and inefficient data processing patterns (e.g., re-tokenizing the same text for different models), which do not fully leverage modern hardware acceleration.
3.  **Developer Experience:** The lack of progress indicators for long operations, scattered resource management, and inconsistent coding standards make it difficult to use, debug, and contribute to.

**1.3. Project Objectives & Key Success Metrics**

| Objective                                         | Key Success Metric(s)                                                                                                    |
| :------------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------- |
| **Achieve full Keras 3 / TF 2.x+ compatibility.**   | All models load and run without errors in a standard Google Colab notebook. The code passes with `tensorflow>=2.16`.     |
| **Optimize inference performance on a T4 GPU.**     | 1. The "Tokenize Once" principle is fully implemented.<br>2. Core prediction logic is compiled with `@tf.function`.<br>3. End-to-end pipeline runtime is measurably faster than a naive, non-optimized port. |
| **Enhance Developer & User Experience.**            | 1. Models and resources are loaded once per session (singleton pattern).<br>2. Resource downloads feature `tqdm` progress bars.<br>3. The codebase is fully type-hinted and documented, adhering to PEP8/257/484. |
| **Deliver a unified, data-centric pipeline.**       | A single `VNLPipeline` class can process a CSV file and output a `pandas.DataFrame` containing all specified linguistic analyses. |
| **Maintain Functional Parity.**                     | The outputs of the refactored models (tags, labels, etc.) are identical to the outputs of the original models for a given set of test sentences. |

**1.4. Scope**

*   **In-Scope:**
    *   Refactoring of `PoSTagger`, `NamedEntityRecognizer`, `DependencyParser`, and `StemmerAnalyzer`.
    *   Integration of `SentimentAnalyzer` and `TreeStack` models.
    *   Creation of a unified `VNLPipeline` for CSV processing.
    *   Modernization of all utility functions and helper scripts.
    *   Ensuring compatibility with a standard Google Colab environment.
*   **Out-of-Scope:**
    *   Training new models or re-training existing ones.
    *   Changing the core algorithms of the models.
    *   Official deployment to production servers (the focus is on the Colab environment).
    *   Creation of a graphical user interface (GUI).

---

### **2. Product Requirements Document (PRD): `vnlp-colab` Pipeline**

**2.1. Overview**

This document specifies the functional and non-functional requirements for the `VNLPipeline` component. The target user is an NLP researcher or developer working with Turkish text data in a pandas-centric workflow.

**2.2. User Stories**

*   **As a Researcher, I want to** process a large CSV file of Turkish sentences **so that I can** receive a structured DataFrame containing token-level POS, NER, dependency, and morphological data for linguistic analysis.
*   **As a Developer, I want to** initialize the pipeline with only the models I need **so that I can** conserve memory and reduce startup time.
*   **As a Colab User, I want to** see progress bars during resource downloads and long processing steps **so that I can** monitor the status of my job and estimate completion time.
*   **As a Contributor, I want to** read well-documented, type-hinted, and modular code **so that I can** easily understand, debug, and extend the library.

**2.3. Functional Requirements**

| ID  | Requirement Description                                                                                                                              |
| :-- | :--------------------------------------------------------------------------------------------------------------------------------------------------- |
| FR1 | The system **shall** provide a `VNLPipeline` class as the main entry point.                                                                        |
| FR2 | The `VNLPipeline` class **shall** be initializable with a list of model names (e.g., `['pos', 'ner']`) to load only the required models.                |
| FR3 | The system **shall** read input data from a tab-separated, headerless `.csv` file with the columns: `t_code, ch_no, p_no, s_no, sentence`.       |
| FR4 | The pipeline **shall** perform the following preprocessing steps in order: sentence cleaning, accent mark removal, and tokenization.                    |
| FR5 | The `PoSTagger` **shall** take a list of tokens and return a simplified list of POS tags (e.g., `['PROPN', 'ADJ', ...]`).                             |
| FR6 | The `NamedEntityRecognizer` **shall** take a list of tokens and return a simplified list of NER tags (e.g., `['O', 'O', 'PER', ...]`).                 |
| FR7 | The `StemmerAnalyzer` **shall** take a list of tokens and return a list of full morphological analyses and a separate list of lemmas.                  |
| FR8 | The `DependencyParser` **shall** process tokens in batches of 40, run parsing, and return a single, combined list of simplified dependency tuples (`(head_index, label)`). |
| FR9 | The final output of the pipeline **shall** be a single `pandas.DataFrame` containing the original data plus columns for all preprocessing steps and model outputs. |
| FR10| The system **shall** save the final DataFrame to a user-specified pickle file.                                                                       |

**2.4. Non-Functional Requirements**

| ID   | Requirement Description                                                                                                                              |
| :--- | :--------------------------------------------------------------------------------------------------------------------------------------------------- |
| NFR1 | **Performance:** Model and resource loading shall be performed only once per session. Subsequent instantiations of the pipeline or models must be near-instantaneous. |
| NFR2 | **Performance:** The core prediction logic of each model must be compiled with `@tf.function` to leverage TensorFlow's graph optimization.         |
| NFR3 | **Environment:** The code must run in a standard Google Colab environment with its default TensorFlow/Keras versions, requiring only `pip` for additional libraries. |
| NFR4 | **Usability:** All long-running operations (downloads, DataFrame processing) must display a `tqdm` progress bar.                                    |
| NFR5 | **Maintainability:** The codebase must adhere to PEP8, PEP257, and PEP484 standards (linting, docstrings, type hints).                              |
| NFR6 | **Licensing:** All generated source code must include the AGPL-3.0 license header.                                                                    |

---

## **Part 2: Technical Documentation**

These documents provide the engineering blueprint for the `vnlp-colab` project, detailing its construction and the rationale behind key decisions.

### **3. Architecture Design Document (ADD): `vnlp-colab`**

**3.1. System Overview**

The `vnlp-colab` architecture is designed as a modular, high-performance pipeline for processing Turkish text. It consists of three primary layers:
1.  **Utility & Resource Layer (`utils_colab.py`):** The foundation that handles all interactions with the file system and network, provides hardware abstraction, and contains shared Keras helper functions.
2.  **Model Layer (`*_colab.py`, `*_utils_colab.py`):** A collection of specialized modules, each responsible for a single NLP task (POS, NER, etc.). Each module contains the model implementation, its singleton factory, and interfaces with the Utility Layer to load its required assets.
3.  **Orchestration Layer (`pipeline_colab.py`):** A single `VNLPipeline` class that acts as the user-facing API. It coordinates the preprocessing of data and the sequential execution of models from the Model Layer.

**3.2. Component Diagram (C4 Model - Level 2)**

```
+-------------------------------------------------------------------------+
| User (in Colab Notebook)                                                |
+-------------------------------------------------------------------------+
      |
      | 1. Calls VNLPipeline.run(csv_path)
      v
+-------------------------------------------------------------------------+
| Orchestration Layer: VNLPipeline (`pipeline_colab.py`)                  |
|-------------------------------------------------------------------------|
| - Reads CSV -> DataFrame                                                |
| - Preprocesses DataFrame (normalize, tokenize)                          |
| - For each row:                                                         |
|   - Calls Stemmer.predict(tokens) -> 'morph', 'lemma'                   |
|   - Calls PoSTagger.predict(tokens) -> 'pos'                            |
|   - Calls NER.predict(tokens) -> 'ner'                                  |
|   - Calls DP.predict(batched_tokens) -> 'dep'                           |
| - Saves final DataFrame                                                 |
+-------------------------------------------------------------------------+
      |
      | 2. Invokes models via Singleton Factories
      v
+-------------------------------------------------------------------------+
| Model Layer (`pos_colab.py`, `ner_colab.py`, etc.)                      |
|-------------------------------------------------------------------------|
| - PoSTagger Factory -> SPUContextPoS Instance                           |
|   - Loads weights, tokenizers                                           |
|   - Uses @tf.function compiled predict()                                |
| - NER Factory -> SPUContextNER Instance                                 |
|   - ... (similar pattern) ...                                           |
| - DP Factory -> SPUContextDP Instance                                   |
|   - ... (similar pattern) ...                                           |
| - Stemmer Factory -> StemmerAnalyzer Instance                           |
|   - ... (similar pattern) ...                                           |
+-------------------------------------------------------------------------+
      |
      | 3. Fetches resources (weights, tokenizers)
      v
+-------------------------------------------------------------------------+
| Utility & Resource Layer (`utils_colab.py`)                             |
|-------------------------------------------------------------------------|
| - download_resource(): Downloads from URL with tqdm to /content/vnlp_cache |
| - load_keras_tokenizer(): Loads tokenizer from JSON                     |
| - detect_hardware_strategy(): Configures TF for GPU/TPU                 |
+-------------------------------------------------------------------------+
```

**3.3. Key Design Patterns**

*   **Singleton Factory:** Used for all major model classes (`PoSTagger`, `NER`, etc.) to prevent redundant loading of heavy models and tokenizers. A global dictionary caches the first instance of each model configuration.
*   **Dependency Injection:** The `Normalizer` class accepts a `StemmerAnalyzer` instance during initialization. This allows the pipeline to create one `StemmerAnalyzer` and share it, preventing the `Normalizer` from creating its own redundant instance.
*   **Facade (`VNLPipeline`):** The `VNLPipeline` class acts as a facade, providing a simple, high-level interface that hides the complexity of the underlying model interactions, data processing, and state management.

---

### **4. Architecture Decision Records (ADRs)**

**ADR-001: Adopt Absolute Imports for Colab Environment**

*   **Context:** When developing Python code to run in a non-packaged environment like Google Colab's `/content/` directory, relative imports (`from .module import X`) fail with an `ImportError: attempted relative import with no known parent package`.
*   **Decision:** We will use **absolute imports** for all intra-project modules (e.g., `from utils_colab import ...`). All refactored `.py` files are expected to reside in the same root directory (`/content/`) or in subdirectories that are added to `sys.path`.
*   **Consequences:**
    *   **Pro:** The code runs directly in Colab without needing to be installed as a package.
    *   **Con:** This pattern is not standard for distributable packages. If the project is formally packaged later, these imports will need to be updated to be relative to the package root (e.g., `from vnlp_colab.utils import ...`).

**ADR-002: Enforce "Tokenize Once" Principle**

*   **Context:** The original library's design allowed each model to perform its own tokenization on a raw sentence string, leading to redundant and inefficient processing in a pipeline.
*   **Decision:** The main `VNLPipeline` will be responsible for tokenizing each sentence exactly once using `TreebankWordTokenize`. The resulting list of tokens will be stored in the DataFrame and passed as an argument to the `predict` method of all token-based models (`PoSTagger`, `NER`, `StemmerAnalyzer`, `DependencyParser`).
*   **Consequences:**
    *   **Pro:** Significant performance improvement by eliminating redundant computation.
    *   **Pro:** Ensures absolute consistency in tokenization across all models.
    *   **Con:** Requires a change in the public signature of each model's `predict` method from `predict(sentence: str)` to `predict(tokens: List[str])`. The `CharNER` model remains an exception due to its unique character-level logic.

**ADR-003: Use Keras 3 Functional API for Model Definition**

*   **Context:** The original models used a mix of legacy Keras/TensorFlow patterns, including `tf.keras.models.Sequential` with an `InputLayer`, which can be brittle and less explicit than the Functional API. Keras 3 strongly favors the Functional API or model subclassing for clarity and robustness.
*   **Decision:** All `create_*_model` utility functions will be rewritten using the **Keras 3 Functional API**. This involves defining all inputs with `keras.Input` and explicitly defining the flow of tensors through the layers.
*   **Consequences:**
    *   **Pro:** The resulting models are fully compatible with Keras 3 and future versions of Keras.
    *   **Pro:** The model graph is explicit and easier to debug and visualize.
    *   **Pro:** It guarantees that the architecture matches the pre-trained weights, as the layer connections are unambiguous.
    *   **Con:** Requires a one-time, careful effort to translate each model architecture from the blueprint.

---

### **5. API Documentation (Generated from Docstrings)**

*(This section would be auto-generated by a tool like Sphinx based on the code's docstrings.)*

#### `VNLPipeline`

`__init__(self, models_to_load: List[str])`
:   Initializes the pipeline and loads the specified models.
:   **Args:**
:       `models_to_load` (List[str]): A list of model names to load. Supported: `['pos', 'ner', 'dep', 'stemmer']`.

`run(self, csv_path: str, output_pickle_path: str) -> pd.DataFrame`
:   Executes the full pipeline: loads data from a CSV, preprocesses it, runs all initialized models, and saves the final enriched DataFrame to a pickle file.
:   **Args:**
:       `csv_path` (str): Path to the input tab-separated CSV file.
:       `output_pickle_path` (str): Path where the final `pandas.DataFrame` will be saved.
:   **Returns:**
:       `pd.DataFrame`: The final DataFrame containing all original and newly generated analysis columns.

#### `PoSTagger`

`predict(self, tokens: List[str]) -> List[Tuple[str, str]]`
:   Predicts Part-of-Speech tags for a pre-tokenized list of words.
:   **Args:**
:       `tokens` (List[str]): A list of string tokens for a single sentence.
:   **Returns:**
:       `List[Tuple[str, str]]`: A list of `(token, pos_tag)` tuples.

*(... and so on for `NamedEntityRecognizer`, `DependencyParser`, and `StemmerAnalyzer`.)*