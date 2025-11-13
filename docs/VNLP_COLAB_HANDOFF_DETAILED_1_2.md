### `VNLP_COLAB_HANDOFF_DETAILED.md` (Part 1/2)

# VNLP Colab Refactor: Detailed Handoff & Technical Blueprint v2.0

**Version:** 2.0
**Date:** 2025-11-13
**Status:** Core `SPUContext` model refactoring is complete. This document addresses key design questions, integrates `TreeStack` and `Sentiment` models into the plan, and outlines a concrete path for formal packaging and testing.

## 1. Core Design Questions & Resolutions

This section clarifies the design choices and issues you raised regarding the refactored code.

#### 1.1. Parameterization: Explicit vs. `**kwargs`

*   **Issue:** You noted changes in how parameters are passed to `create_stemmer_model` and `process_stemmer_input` (from explicit arguments to `**self.params`). You questioned if being more "open" (explicit) is better.
*   **Background:** The choice between `create_model(**params)` and `create_model(arg1=val1, ...)` is a trade-off between conciseness and clarity. The `**params` style is common when a configuration dictionary drives model creation, keeping the call site clean. The explicit style makes the function's dependencies immediately obvious without needing to look up the `params` dictionary.
*   **Resolution & Decision:** You are correct; **explicitness is better for maintainability and clarity.** The code relies on a precise mapping between the keys in the `params` dictionary and the function arguments. Using `**self.params` hides this dependency and can lead to runtime errors if a key is renamed.
*   **Action for Next Session:** We will revert to passing all parameters explicitly in `create_stemmer_model` and `process_stemmer_input`. This makes the code more robust and easier for new developers to understand and modify.

#### 1.2. The `shuffle` Argument in `process_stemmer_input`

*   **Issue:** You asked about the purpose of the `shuffle` argument and why it's not used in inference.
*   **Background:** The `shuffle` argument was originally used **only during model training**. For the Stemmer/Disambiguator, each token has multiple analysis candidates (e.g., `['analysis_A', 'analysis_B', 'analysis_C']`), where only the first one is correct. During training, shuffling these candidates (`['analysis_C', 'analysis_A', 'analysis_B']`) along with their corresponding labels (`[0, 1, 0]`) creates a more challenging and robust learning task. It prevents the model from simply learning that the correct answer is always at index 0.
*   **Resolution & Decision:** During **inference (prediction)**, we must *not* shuffle the candidates. The model's job is to identify the single correct analysis from the fixed list provided by the `TurkishStemSuffixCandidateGenerator`. The note "not used for inference" is correct and serves as an important clarification. No code change is needed, but this context is vital.

#### 1.3. Standalone Testing for Individual Modules

*   **Issue:** You correctly pointed out that the `if __name__ == "__main__"` block was removed from the final `ner_colab.py` and other modules, which is undesirable as it prevents individual testing.
*   **Background:** In the process of creating a unified pipeline, the focus shifted away from module-level execution. This was a mistake, as standalone testing is critical for debugging and modular development.
*   **Resolution & Decision:** We will restore a `main()` function and an `if __name__ == "__main__"` block to each primary model file (`pos_colab.py`, `ner_colab.py`, `dep_colab.py`, `stemmer_colab.py`). This will allow you to run `python -m pos_colab` from the terminal (or execute the file in a similar manner) to test just that component with a sample sentence.

## 2. Integrating Missing and Advanced Components

The initial refactoring focused on the `SPUContext` models. We will now formally integrate the remaining models and features into the plan.

#### 2.1. `TreeStack` Model Integration

*   **Issue:** The `TreeStack` models for PoS and DP were noted as "future work." You have confirmed they are a requirement.
*   **Background:** The `TreeStack` models are more complex because they have a **dependency chain**. `TreeStackDP` requires the output of `TreeStackPoS`, which in turn requires the output of `StemmerAnalyzer`. This makes them fundamentally different from the `SPUContext` models that operate independently on tokenized text.
*   **Technical Implementation Plan:**
    1.  **Refactor `_treestack_utils.py`:** Create `pos_treestack_utils_colab.py` and `dep_treestack_utils_colab.py`. Rewrite `create_pos_tagger_model` and `create_dependency_parser_model` using the Keras 3 Functional API, precisely matching their original architectures.
    2.  **Create `TreeStackPoS` and `TreeStackDP` Classes:** In `pos_colab.py` and `dep_colab.py`, create these new classes. Their `__init__` methods will require other model instances to be passed in (e.g., `TreeStackPoS` will take a `StemmerAnalyzer` instance).
    3.  **Update Singleton Factories:** The `PoSTagger` and `DependencyParser` factory classes will be updated to handle requests for `TreeStackPoS` and `TreeStackDP`, managing the dependency injection.
    4.  **Update `VNLPipeline`:** The pipeline's `run_analysis` method will be modified to execute models in the correct order if a `TreeStack` model is requested: **Stemmer -> PoS -> DP**.

#### 2.2. `SentimentAnalyzer` Integration

*   **Issue:** The `SentimentAnalyzer` was omitted from the final pipeline. You have confirmed it should be included.
*   **Background:** The `SentimentAnalyzer` is simpler than the other models. It operates on the full sentence string (specifically, the `no_accents` version) and does not require pre-tokenization in the same way as the other models.
*   **Technical Implementation Plan:**
    1.  Create `sentiment_utils_colab.py` with a Keras 3 compliant `create_spucbigru_sentiment_model`.
    2.  Create `sentiment_colab.py` with the `SPUCBiGRUSentimentAnalyzer` class and a `SentimentAnalyzer` factory. This class's `predict` method will accept a sentence string.
    3.  In `VNLPipeline`, if `'sentiment'` is requested, load the model. In `run_analysis`, apply the model's `predict_proba` method to the `no_accents` column and store the result in a new `sentiment` column. This is an optimal point as it requires no token-level processing.