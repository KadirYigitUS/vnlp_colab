### `VNLP_COLAB_HANDOFF.md` (Part 2/2)

## 3. Final Code Structure & Technical Blueprint

The final refactored package is designed to be run as a collection of scripts within the `/content/` directory of a Google Colab notebook. The structure is as follows:

```
/content/
├── utils_colab.py            # Core: Downloading, logging, hardware, base Keras helpers
├── tokenizer_colab.py        # Tokenization functions (TreebankWordTokenize)
|
├── normalizer/
│   ├── normalizer_colab.py   # Full-featured Normalizer class
│   └── _deasciifier.py       # Dependency for the Normalizer
|
├── pos_utils_colab.py        # PoS model architecture and preprocessing
├── pos_colab.py              # Main PoSTagger class and singleton factory
|
├── ner_utils_colab.py        # NER model architectures and preprocessing (consolidated)
├── ner_colab.py              # Main NamedEntityRecognizer class and factory
|
├── dep_utils_colab.py        # DP model architecture and preprocessing (consolidated)
├── dep_colab.py              # Main DependencyParser class and factory
|
├── stemmer_utils_colab.py    # Stemmer model architecture and preprocessing
├── stemmer_colab.py          # Main StemmerAnalyzer class and singleton factory
|
└── pipeline_colab.py         # Main entry point: VNLPipeline class orchestrator
```

### Technical Workflow:

1.  **Setup:** A user's Colab notebook will first `!pip install` the required dependencies (`sentencepiece`, `spylls`, etc.).
2.  **Initialization:** The user imports and instantiates `VNLPipeline`, passing a list of models to load (e.g., `['pos', 'ner', 'dep', 'stemmer']`).
    *   The `VNLPipeline` constructor calls the singleton factory for each requested model (`PoSTagger()`, `NamedEntityRecognizer()`, etc.).
    *   On first call, each model class downloads its weights and resources to `/content/vnlp_cache` and compiles its prediction function using `@tf.function`.
    *   Subsequent calls to instantiate these classes will be instantaneous, returning the cached instance.
3.  **Execution:** The user calls `pipeline.run(csv_path, output_path)`.
    *   The pipeline reads the input CSV into a pandas DataFrame.
    *   `run_preprocessing` is executed, applying cleaning, normalization, and tokenization in a vectorized manner using `df.progress_apply`.
    *   `run_analysis` is executed. For each row in the DataFrame, it passes the pre-tokenized list to each model's `.predict()` method.
    *   The results from each model are simplified as per the requirements (e.g., extracting only the tag from PoS tuples) and stored in new DataFrame columns.
    *   The final, enriched DataFrame is saved to a pickle file.

## 4. Future Considerations & Potential Improvements

While the core refactoring is complete, several opportunities exist for further enhancement.

1.  **Formal Python Package:**
    *   **Consideration:** Currently, the code runs as a collection of scripts. For easier distribution and dependency management, this could be structured into a formal Python package with a `setup.py` or `pyproject.toml`.
    *   **Action:** Create a `vnlp_colab/` directory, place all scripts inside it, add `__init__.py` files to each subdirectory, and write a `setup.py`. This would allow users to `pip install git+https://your-repo/vnlp_colab.git`.

2.  **`TreeStack` Model Integration:**
    *   **Consideration:** The current refactoring focused on the `SPUContext` models. The `TreeStack` models for `PoSTagger` and `DependencyParser` have more complex dependencies (they depend on the `StemmerAnalyzer` and `PoSTagger` outputs).
    *   **Action:** Refactor `_treestack_utils.py` for both PoS and DP using the same Keras 3 Functional API pattern. Integrate them into their respective factory classes (`pos_colab.py`, `dep_colab.py`) and update the `VNLPipeline` to handle their unique dependency chain (i.e., ensure Stemmer and PoS run *before* TreeStackDP).

3.  **Sentiment Analyzer:**
    *   **Consideration:** The `SentimentAnalyzer` was mentioned but not included in the final pipeline model list.
    *   **Action:** Create `sentiment_colab.py` and `sentiment_utils_colab.py` following the same refactoring pattern used for the other models. Add `'sentiment'` as a valid model key in the `VNLPipeline`.

4.  **Advanced Performance Tuning:**
    *   **Mixed Precision:** For users with Tensor Core GPUs (T4, V100, A100), enabling mixed precision (`keras.mixed_precision.set_global_policy('mixed_float16')`) could further accelerate inference with minimal accuracy loss. This could be an optional setting in `utils_colab.py`.
    *   **`tf.data` Pipeline:** For extremely large datasets that do not fit in memory, the pandas-based pipeline could be converted to a `tf.data.Dataset` pipeline. This would allow for streaming data from disk and applying transformations on the fly, offering better memory management and performance at the cost of increased code complexity.

5.  **Testing and Validation:**
    *   **Consideration:** The current testing is done via a `main` block. A formal testing suite would improve robustness.
    *   **Action:** Create a `tests/` directory with `pytest` scripts. Each test could load a small fixture, run a specific model or the full pipeline, and assert that the output matches an expected result. This is crucial for verifying correctness after any future changes.
