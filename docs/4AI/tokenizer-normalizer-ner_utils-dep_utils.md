# tokenizer-normalizer-ner_utils-dep_utils.md

### Final Holistic Debug Summary

Here are the key adjustments and fixes made during the final review:

1.  **"Tokenize Once" Principle Implemented:** The `predict` methods for `PoSTagger`, `NamedEntityRecognizer`, `DependencyParser`, and `StemmerAnalyzer` have been modified. They now accept a list of pre-tokenized strings (`List[str]`) instead of a raw sentence string. This is a major performance optimization, as the pipeline will now tokenize each sentence only once.

2.  **Consolidation of Utilities:** To simplify the file structure in Colab, several smaller utility files have been consolidated:
    *   `tokenizer.py` logic is now in a new `tokenizer_colab.py`.
    *   The `Normalizer` class and its dependencies are now in `normalizer_colab.py`.
    *   `ner_to_displacy_format` and `create_charner_model` have been moved into `ner_utils_colab.py`.
    *   `decode_arc_label_vector` and `dp_pos_to_displacy_format` have been moved into `dep_utils_colab.py`.
    This reduces the number of files you need to manage in `/content/`.

3.  **Corrected Resource URLs:** All placeholder URLs for downloading resources have been replaced with the direct GitHub raw links you provided.

4.  **Pipeline Logic Refined:** The `pipeline_colab.py` script has been updated to pass the `tokens` column (the list of tokens) to the models, aligning with the "tokenize once" strategy.