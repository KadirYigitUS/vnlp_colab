# `stemmer_colab.py`

### Holistic Debug and Review

1.  **File Naming:** `stemmer_colab.py` is used.
2.  **Imports:** All imports correctly point to the new `_colab` utility modules. The `TurkishStemSuffixCandidateGenerator` is now accessed via its own singleton factory (`get_candidate_generator_instance`) for added efficiency.
3.  **Singleton Pattern:** The module provides a `get_stemmer_analyzer` factory function, which is the standard way to implement a singleton for a class. This ensures that the heavyweight `StemmerAnalyzer` is only instantiated once.
4.  **Keras 3 Compliance and Performance:**
    *   The class correctly uses `create_stemmer_model` from `stemmer_utils_colab.py`.
    *   The prediction logic is batched, which is critical for performance. It correctly handles sentences longer than the batch size by iterating in chunks.
    *   The core model call is wrapped in `@tf.function` (`compiled_predict_step`), ensuring graph compilation.
    *   The post-prediction masking logic is efficient, using `tf.sequence_mask` and vectorized `argmax`.
5.  **Data Processing:** The class correctly calls the optimized `process_stemmer_input` function. It also includes the crucial fix for the Keras tag tokenizer (`split=' '`, `filters=''`) to handle tag sequences correctly.
6.  **Configuration Management:** All hyperparameters and resource links are neatly organized in the `_MODEL_CONFIGS` dictionary, making the code clean and easy to maintain.