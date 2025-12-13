# VNLP Colab Refactor: Detailed Handoff & Technical Blueprint v2.0

**Version:** 2.0
**Date:** 2025-11-13
**Status:** All refactoring, integration, packaging, and testing phases are complete. The `vnlp-colab` package is ready for distribution and use.

# 1. Project Goals & Accomplishments

The primary objective was to refactor the legacy `vnlp` Python package for high-performance, maintainable, and user-friendly execution in a standard Google Colab environment.

### Key Accomplishments:

1.  **Full Keras 3 & TensorFlow 2.x+ Compatibility:** All model architectures were rewritten using the modern Keras 3 Functional API, ensuring future compatibility and correct loading of pre-trained weights.
2.  **Performance Optimization for Colab:**
    *   **Graph Compilation:** Core prediction logic is compiled with `@tf.function` for a significant speed boost.
    *   **Singleton Pattern:** All models are loaded into memory only once per session, making subsequent initializations instantaneous.
    *   **"Tokenize Once" Principle:** The main pipeline tokenizes each sentence only once, eliminating redundant preprocessing steps.
3.  **Full Model Integration:** All specified `SPUContext`, `TreeStack`, and `SentimentAnalyzer` models have been successfully refactored and integrated. The pipeline automatically handles the complex dependency chain of the `TreeStack` models.
4.  **Enhanced Developer Experience:** The codebase now features a central utility module, `tqdm` progress bars for all long-running operations, structured logging instead of `print()` statements, and full adherence to modern Python standards (type hinting, docstrings, `pathlib`).
5.  **Formal Python Package:** The project has been structured as a formal, installable Python package with a `pyproject.toml` file, enabling simple installation directly from GitHub.
6.  **Automated Testing Suite:** A `pytest`-based suite provides unit tests for individual components and an integration test for the end-to-end pipeline, ensuring correctness and preventing future regressions.

## 2. Architecture Design Document (ADD): `vnlp-colab`

### 2.1. System Overview

The `vnlp-colab` architecture is designed as a modular, high-performance pipeline. It consists of three primary layers:
1.  **Utility & Resource Layer (`vnlp_colab/utils_colab.py`):** The foundation that handles networking, caching, hardware abstraction, and shared Keras helpers.
2.  **Model Layer (e.g., `vnlp_colab/pos/`):** A collection of specialized modules for each NLP task. Each module contains the model implementation(s), a singleton factory, and interfaces with the Utility Layer.
3.  **Orchestration Layer (`vnlp_colab/pipeline_colab.py`):** The user-facing `VNLPipeline` class that coordinates data flow and model execution.

### 3.2. Key Design Patterns

*   **Singleton Factory:** Used for all major model classes (`PoSTagger`, `NamedEntityRecognizer`, etc.) to prevent redundant loading.
*   **Dependency Injection:** The `VNLPipeline` and model factories automatically resolve and inject dependencies (e.g., passing a `StemmerAnalyzer` instance to `TreeStackPoS`).
*   **Facade:** The `VNLPipeline` class provides a simple, high-level interface that hides the complexity of the underlying model interactions and data processing.

## 4. Architecture Decision Records (ADRs)

**ADR-001: Adopt Package-Relative Imports for Formal Packaging**

*   **Context:** To create a distributable Python package, imports must be resolved within the package itself, not based on the script's location in a flat directory.
*   **Decision:** All intra-project imports were converted to **package-relative imports** (e.g., `from vnlp_colab.utils_colab import ...`).
*   **Consequences:** The code is now a standard, installable Python package. It can no longer be run as a collection of separate scripts in `/content/` without installation, which is the desired final state for distribution.

**ADR-002: Enforce "Tokenize Once" Principle**

*   **Context:** The original library's design allowed each model to perform its own tokenization, leading to redundant and inefficient processing.
*   **Decision:** The main `VNLPipeline` is responsible for tokenizing each sentence exactly once. The resulting list of tokens is passed to the `predict` method of all token-based models.
*   **Consequences:** Significant performance improvement and guaranteed tokenization consistency across all models.

**ADR-003: Use Keras 3 Functional API for Model Definition**

*   **Context:** The original models used legacy Keras patterns. Keras 3 strongly favors the Functional API for clarity and robustness, especially for complex, multi-input models.
*   **Decision:** All `create_*_model` utility functions were rewritten using the **Keras 3 Functional API**, precisely matching the original blueprints to ensure weight compatibility.
*   **Consequences:** The models are fully compatible with Keras 3+, and their architecture is explicit and easier to debug.

## 3. Future Considerations & Next Steps

1.  **Advanced Performance Tuning (ONNX):** The next major performance enhancement would be to convert the Keras models to the **ONNX (Open Neural Network Exchange)** format. Using an inference engine like `onnxruntime-gpu` can often provide a significant speedup over native TensorFlow for inference, with no accuracy loss. This is the recommended next step for performance optimization.
2.  **`tf.data` Pipeline for Large Datasets:** For datasets that are too large to fit into memory as a pandas DataFrame, the pipeline could be re-architected to use a `tf.data.Dataset` pipeline. This would enable streaming data from disk, offering superior memory management at the cost of increased code complexity.
3.  **Mixed Precision:** An option to enable mixed-precision (`mixed_float16`) can be added to the `VNLPipeline` for users with modern GPUs (T4, V100, A100) to potentially further accelerate inference.


# Current Colab Code: [running on T4GPU]
https://colab.research.google.com/drive/1rKb-4PfN5DsXOdK9zbtxpou5mtimbfFf?usp=sharing

# --- Step 1: Install Dependencies ---
print("Installing required libraries...")
!pip install sentencepiece==0.2.1 spylls tqdm regex -q
!pip install -q git+https://github.com/KadirYigitUS/vnlp_colab.git #github address
from vnlp_colab.pipeline_colab import VNLPipeline
import pandas as pd

# Create a dummy CSV file for demonstration
dummy_data = (
    "novel01\t1\t1\t1\tBu film harikaydı, çok beğendim.\n"
    "novel01\t1\t1\t2\tBenim adım Melikşah ve İstanbul'da yaşıyorum.\n"
)
with open("/content/input.csv", "w", encoding="utf-8") as f:
    f.write(dummy_data)

# Initialize the pipeline with the models you need
# Example: Using default SPUContext models + Sentiment
pipeline = VNLPipeline(models_to_load=['pos', 'ner', 'dep', 'stemmer', 'sentiment'])

# Run the full analysis
final_df = pipeline.run(
    csv_path="/content/input.csv",
    output_pickle_path="/content/analysis_results.pkl"
)

# Display the results
pd.set_option('display.max_columns', None)
print(final_df.head())

Current Error:
Downloading Stemmer_Shen_prod.weights: 100%
 9.32M/9.32M [00:01<00:00, 12.4MiB/s]
Downloading Stemmer_char_tokenizer.json: 
 8.42k/? [00:00<00:00, 763kiB/s]
WARNING:vnlp_colab.utils_colab:Download of 'Stemmer_char_tokenizer.json' might be incomplete. Expected 2641 bytes, got 8424 bytes.
Downloading Stemmer_morph_tag_tokenizer.json: 
 10.9k/? [00:00<00:00, 1.24MiB/s]
WARNING:vnlp_colab.utils_colab:Download of 'Stemmer_morph_tag_tokenizer.json' might be incomplete. Expected 3118 bytes, got 10883 bytes.
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipython-input-2765716488.py in <cell line: 0>()
     12 # Initialize the pipeline with the models you need
     13 # Example: Using default SPUContext models + Sentiment
---> 14 pipeline = VNLPipeline(models_to_load=['pos', 'ner', 'dep', 'stemmer', 'sentiment'])
     15 
     16 # Run the full analysis

5 frames
/usr/local/lib/python3.12/dist-packages/vnlp_colab/stemmer/_yildiz_analyzer.py in _read_suffix_dic(self)
     66     def _read_suffix_dic(self):
     67         dic = {}
---> 68         with open(self.SUFFIX_DICT_FILE_PATH, "r", encoding="UTF-8") as f:
     69             for line in f:
     70                 suffix, tag = line.strip().split("\t")

Current Solution and Issue:
I added the Resources Directories, but they are not properly referred in the scripts

Tasks:
1- Resolve resources issues, some are downloaded from original github repo of VNLP at VNGRS-AI, but if they are in resources dirs, we shouldn't download them. Similarly 'Stemmer_char_tokenizer.json' shows mismatch, load them from resources dir.

2- treestack_pos.py (vnlp-custom) has the following feature
        # I don't want StemmerAnalyzer to occupy any memory in GPU!
        if stemmer_analyzer is None:
            with tf.device('/cpu:0'):
                stemmer_analyzer = StemmerAnalyzer()
        self.stemmer_analyzer = stemmer_analyzer
Can we or should we use similar memory freeing methods

3- dep_colab.py lacks displacy format? Below is from previous version of the code
    def predict(
        self,
        sentence: str,
        tokens: List[str],
        displacy_format: bool = False,
        pos_result: List[Tuple[str, str]] = None
    ) -> List[Tuple[int, str, int, str]]:
        """
        High-level API for Sentence Dependency Parsing.

        Args:
            sentence (string): Input Sentence (Not used for parsing, but for displacy_format)
            tokens (list): Input sentence tokens.
        Returns:
            List[Tuple[str, str]]: A list of tuple of tokens and DEP Parsing results 
        """
        if not tokens:
            return []

        sentence_max_len = self.params['sentence_max_len']
        if len(tokens) > sentence_max_len:
            logger.warning(f"Token count ({len(tokens)}) exceeds max length ({sentence_max_len}). Truncating.")
            tokens = tokens[:sentence_max_len]
        
        num_tokens = len(tokens)
        arcs: List[int] = []
        labels: List[int] = []

        for t in range(num_tokens):
            inputs_np = process_dp_input(
                t, tokens, self.spu_tokenizer_word, self.tokenizer_label,
                self.arc_label_vector_len, arcs, labels
            )
            inputs_tf = [tf.convert_to_tensor(arr) for arr in inputs_np]
            logits = self.compiled_predict_step(*inputs_tf).numpy()[0]
            
            arc, label = decode_arc_label_vector(logits, sentence_max_len, self.label_vocab_size)
            arcs.append(arc)
            labels.append(label)

        dp_result = [
            (idx + 1, word, int(arcs[idx]), self.tokenizer_label.sequences_to_texts([[labels[idx]]])[0] or "UNK")
            for idx, word in enumerate(tokens)
        ]

        return dp_pos_to_displacy_format(dp_result, pos_result) if displacy_format else dp_result

4- normalizer.py: _load_dictionary and other hunspell were previously used for currently defunct correct_typos function; but now there is no need for them we can clear operations related with correct_typos. Just leave a not that it is defunct.