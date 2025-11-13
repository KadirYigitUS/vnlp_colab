### **Technical Blueprint: Packaging and Testing Suite (Phase 3 & 4)**

#### **1. Objective**

The goal is to transition the collection of refactored scripts into a formal, installable Python package and to create a `pytest`-based suite for unit and integration testing. This will ensure the project is robust, maintainable, and easily distributable for Colab users.

#### **2. Final Project Structure**

The project will be organized into the following formal package structure. This layout separates the source code from the tests and includes the necessary packaging configuration file.

```
vnlp-colab/
├── pyproject.toml
├── tests/
│   ├── __init__.py
│   ├── test_dep.py
│   ├── test_ner.py
│   ├── test_pipeline.py
│   ├── test_pos.py
│   └── test_stemmer.py
└── vnlp_colab/
    ├── __init__.py
    ├── dep/
    │   ├── __init__.py
    │   ├── dep_colab.py
    │   ├── dep_treestack_utils_colab.py
    │   └── dep_utils_colab.py
    ├── ner/
    │   ├── __init__.py
    │   ├── ner_colab.py
    │   └── ner_utils_colab.py
    ├── normalizer/
    │   ├── __init__.py
    │   ├── _deasciifier.py
    │   └── normalizer_colab.py
    ├── pipeline_colab.py
    ├── pos/
    │   ├── __init__.py
    │   ├── pos_colab.py
    │   ├── pos_treestack_utils_colab.py
    │   └── pos_utils_colab.py
    ├── sentiment/
    │   ├── __init__.py
    │   ├── sentiment_colab.py
    │   └── sentiment_utils_colab.py
    ├── stemmer/
    │   ├── __init__.py
    │   ├── _yildiz_analyzer.py
    │   ├── stemmer_colab.py
    │   └── stemmer_utils_colab.py
    ├── tokenizer_colab.py
    └── utils_colab.py
```

#### **3. Step-by-Step Implementation Plan**

**Step 3.1: Formal Packaging (`pyproject.toml`)**

*   **New File:** `pyproject.toml`
*   **Action:** I will generate a complete `pyproject.toml` file. This file is the modern standard for Python packaging and will define:
    *   **Project Metadata:** `name`, `version`, `authors`, `description`, `license` (AGPL-3.0).
    *   **Build System:** Specifies `setuptools` as the build backend.
    *   **Dependencies:** Lists all required libraries (`tensorflow`, `pandas`, `sentencepiece==0.2.1`, `spylls`, etc.) for automatic installation via `pip`.
*   **Code Modification:** All `import` statements within the source code will be updated to be relative to the `vnlp_colab` package root (e.g., `from vnlp_colab.utils_colab import ...`). This makes the code a proper, relocatable package.

**Step 3.2: Create the Testing Suite (`tests/`)**

*   **New Directory & Files:** `tests/test_*.py`
*   **Actions:**
    1.  **`tests/test_pipeline.py` (Integration Test):**
        *   This test will create a small, in-memory CSV-like fixture with 2-3 sentences.
        *   It will initialize the `VNLPipeline` with a full set of models.
        *   It will run the full pipeline on the fixture.
        *   It will assert that the output is a `pandas.DataFrame` and contains all the expected analysis columns (`pos`, `ner`, `dep`, `lemma`, `sentiment`, etc.).
    2.  **`tests/test_pos.py` (Unit Test):**
        *   This test will import and initialize the `PoSTagger` factory.
        *   It will run `predict()` on a simple, pre-tokenized list of words.
        *   It will assert that the output has the correct length and that specific key tokens have the correct predicted PoS tag.
    3.  **`tests/test_ner.py`, `tests/test_dep.py`, `tests/test_stemmer.py` (Unit Tests):**
        *   These tests will follow the same pattern as `test_pos.py`, each focusing on a single component to verify its core `predict` functionality and output format.

#### **4. Deliverables for this Phase**

1.  The complete `pyproject.toml` file.
2.  The complete script for each test file:
    *   `test_pipeline.py`
    *   `test_pos.py`
    *   `test_ner.py`
    *   `test_dep.py`
    *   `test_stemmer.py`
3.  All previously generated `*_colab.py` and `*_utils_colab.py` files will have their import statements updated to reflect the new package structure.

---

This blueprint finalizes the project by adding the crucial layers of packaging and automated testing. I will now await your approval to proceed with generating the `pyproject.toml` file.