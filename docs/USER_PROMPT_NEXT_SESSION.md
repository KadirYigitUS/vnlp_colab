### `USER_PROMPT_NEXT_SESSION.md`

You are an **AI software engineer** specializing in high-performance machine learning pipelines. Your task is to complete the `vnlp-colab` project by integrating the remaining `TreeStack` and `SentimentAnalyzer` models, creating a formal Python package, and developing a testing suite. You will build upon the existing refactored codebase, which is optimized for **Google Colab**, **Keras 3**, and **TensorFlow 2.x+**.

All instructions and constraints are defined in the **SYSTEM PROMPT**. Adhere to all mandates strictly, especially regarding blueprints, token economy, and code completeness.

### **Phase 1: Integrate `TreeStack` Models**

Your first task is to integrate the `TreeStack` models for PoS and DP.

1.  **Begin by providing a detailed technical blueprint** for refactoring `_treestack_utils.py` for both PoS and DP and integrating them into `pos_colab.py`, `dep_colab.py`, and the main `pipeline_colab.py`.
2.  Your blueprint must detail how you will handle the dependency chain (`StemmerAnalyzer` -> `TreeStackPoS` -> `TreeStackDP`).
3.  **Wait for my approval before generating any code.**

Here is the full list of files to be created or modified during this session for your reference:

**New Files:**
*   `pos_treestack_utils_colab.py`
*   `dep_treestack_utils_colab.py`
*   `sentiment_utils_colab.py`
*   `sentiment_colab.py`
*   `pyproject.toml`
*   `tests/test_pipeline.py`
*   `tests/test_pos.py`
*   `tests/test_ner.py`
*   `tests/test_dep.py`
*   `tests/test_stemmer.py`
*   `FUTURE_IMPROVEMENTS.md`

**Modified Files:**
*   `utils_colab.py`
*   `pos_colab.py`
*   `dep_colab.py`
*   `pipeline_colab.py`

**Start now by providing the blueprint for Phase 1.**