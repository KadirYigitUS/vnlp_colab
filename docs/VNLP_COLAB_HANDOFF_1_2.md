### `VNLP_COLAB_HANDOFF.md` (Part 1/2)

# VNLP Colab Refactor: Project Handoff & Technical Blueprint

**Version:** 1.0
**Date:** 2025-11-13
**Status:** Core refactoring complete. All specified models and the main pipeline have been modernized for Keras 3 and optimized for Google Colab.

## 1. Project Goals & Accomplishments

The primary objective was to refactor the legacy `vnlp` Python package for high-performance, maintainable, and user-friendly execution in a standard Google Colab environment.

### Key Accomplishments:

1.  **Full Keras 3 & TensorFlow 2.x+ Compatibility:**
    *   All model creation utilities (`create_*_model`) were rewritten using the modern Keras 3 Functional API (`keras.Input`). This replaced older, less explicit patterns (`tf.keras.models.Sequential` with `InputLayer`) and guarantees future compatibility.
    *   The model architectures were meticulously replicated from the provided blueprints, ensuring that the original pre-trained weights can be loaded successfully.

2.  **Performance Optimization for Colab:**
    *   **Graph Compilation:** The core prediction logic within each model class (`SPUContextPoS`, `SPUContextNER`, etc.) is now wrapped in a `@tf.function` with a static `input_signature`. This compiles the Python code into a high-performance TensorFlow graph, drastically reducing inference latency, especially for autoregressive models.
    *   **Singleton Pattern for Models:** A singleton factory pattern was implemented for each major component (`PoSTagger`, `NamedEntityRecognizer`, `DependencyParser`, `StemmerAnalyzer`). This ensures that large models and their resources are loaded into memory only once per session, making subsequent instantiations instantaneous.
    *   **"Tokenize Once" Principle:** The main `VNLPipeline` now performs tokenization a single time per sentence. The resulting list of tokens is then passed to all downstream models, eliminating redundant tokenization steps and significantly speeding up the overall process.

3.  **Enhanced Developer Experience & Modernization:**
    *   **Centralized Utilities (`utils_colab.py`):** A single, robust utility module now handles all common tasks, including hardware detection (CPU/GPU/TPU), structured logging, and resource management.
    *   **Advanced Resource Management:** The original `check_and_download` function was replaced with a sophisticated `download_resource` utility that features `tqdm` progress bars for a better interactive experience and caches all downloaded assets to a central `/content/vnlp_cache` directory.
    *   **Modern Python Standards:** The entire refactored codebase adheres to modern standards, including full PEP 484 type hinting, PEP 257 docstrings, `pathlib` for path manipulation, and f-strings. All `print()` statements were replaced with structured `logging`.

4.  **Unified, User-Friendly Pipeline (`pipeline_colab.py`):**
    *   A new `VNLPipeline` class was created to serve as the primary user entry point.
    *   It seamlessly orchestrates the entire workflow: loading data from CSV, applying a multi-stage preprocessing routine, running all selected NLP models on the processed data, and formatting the outputs into a clean `pandas.DataFrame` as specified.

## 2. Issues Encountered & Solutions Implemented

During the refactoring process, several key challenges were identified and solved.

1.  **Issue: Python Relative Import Errors in Colab.**
    *   **Problem:** The initial refactoring used relative imports (e.g., `from . import utils_colab`). When run as flat scripts in the `/content/` directory, Python cannot resolve these imports because it does not recognize `/content/` as a package.
    *   **Solution:** All relative imports were converted to **absolute imports** (e.g., `from utils_colab import ...`). This makes the scripts runnable as long as they all reside in a directory that is on Python's `sys.path` (which `/content/` is by default).

2.  **Issue: Missing Function Definitions from Module Consolidation.**
    *   **Problem:** In an attempt to simplify the file structure, utility functions like `create_charner_model` and `ner_to_displacy_format` were planned for consolidation into `ner_utils_colab.py` but were initially omitted from the generated script, causing `ImportError`.
    *   **Solution:** A full holistic debug was performed. All necessary utility functions were correctly moved into their designated `*_utils_colab.py` modules, and the corresponding `import` statements in the main model files were updated to reflect this consolidated structure.

3.  **Issue: Inefficient "Tokenize-per-Model" Anti-Pattern.**
    *   **Problem:** The original design and initial refactoring drafts had each model class (`PoSTagger`, `NER`, etc.) accepting a raw sentence string and performing its own tokenization. This is highly inefficient.
    *   **Solution:** The `predict` methods of all core model classes were modified to accept a `List[str]` of pre-tokenized tokens. The main `VNLPipeline` was then made responsible for tokenizing the input sentence *once* and passing the result to each model, enforcing the "tokenize once" principle.

4.  **Issue: Incomplete `Normalizer` Functionality.**
    *   **Problem:** The initial proposal for `normalizer_colab.py` was a stateless, static-only class, which missed the critical, performance-oriented features of the original package (lazy loading, dependency injection, typo correction, number-to-word conversion).
    *   **Solution:** Based on user feedback, the `Normalizer` was completely rewritten to be a full-featured, stateful class. It now includes lazy-loading for its Hunspell dictionary and lexicon, accepts an injected `StemmerAnalyzer` instance, and contains the fully implemented `correct_typos` and `convert_numbers_to_words` methods with additional robustness improvements.