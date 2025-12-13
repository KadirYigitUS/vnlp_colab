### `README.md` — Project Documentation

**🔧 Summary of Changes:**
*   **Installation Instructions:** Updated to highlight the critical "Restart Runtime" step required after installing dependencies (due to `protobuf` updates).
*   **Pipeline Usage:** Reflected the new `VNLPipeline` API and the `batch_size` parameter.
*   **Automation Guide:** Added a dedicated section for the `COLAB_VNLP_automation_pipeline_2.py` workflow, explaining the Drive directory structure and the benefits of the new mirroring/zipping strategy.
*   **Architecture Specs:** Updated to mention the `tf.data` streaming and "Tokenize Once" optimizations.

# VNLP Colab: High-Performance Turkish NLP Pipeline

**VNLP Colab** is a refactored, optimized version of the [VNLP library](https://github.com/vngrs-ai/vnlp), engineered specifically for **Google Colab**, **Keras 3**, and **TensorFlow 2.x**.

It addresses legacy compatibility issues and introduces a high-throughput **Batch Inference Pipeline** designed to maximize T4 GPU utilization.

---

## 🚀 Key Features

*   **Google Colab Native:** Fixes dependency conflicts (`protobuf`, `sentencepiece`) inherent to the 2025 Colab runtime.
*   **GPU Acceleration:** Replaces row-by-row processing with `tf.data` streaming and `@tf.function` graph compilation.
*   **Tokenize Once:** Pre-tokenizes text a single time per pipeline run, eliminating redundant computation across models.
*   **Smart Batching:** 
    *   Automatically handles dependency parsing limits (sentences > 40 tokens) via internal chunking.
    *   Sorts data by length to minimize padding overhead.
*   **Robust Automation:** Includes a production-grade ETL script for processing thousands of CSVs from Google Drive with auto-recovery and memory management.

---

## 📦 Installation

Run the following in a Colab cell. 


```bash
!pip install -q git+https://github.com/KadirYigitUS/vnlp_colab.git
```

---

## ⚡ Quick Start

### 1. Interactive Pipeline
Use this for analyzing dataframes or single files interactively.

```python
from vnlp_colab.pipeline_colab import VNLPipeline
import pandas as pd

# 1. Initialize (Loads models once using Singleton pattern)
# Available: 'pos', 'ner', 'dep', 'stemmer', 'sentiment'
pipeline = VNLPipeline(models_to_load=['pos', 'ner', 'dep', 'sentiment'])

# 2. Run Analysis
# Input CSV must be tab-separated: t_code, ch_no, p_no, s_no, sentence
final_df = pipeline.run(
    csv_path="/content/input.csv",
    output_pickle_path="/content/output.pkl",
    batch_size=64  # Optimized for T4 GPU
)

# 3. View Results
print(final_df[['tokens', 'pos', 'ner', 'dep']].head())
```

---

## 🏭 Batch Automation (Production)

For processing large archives of CSVs from Google Drive, use an automation script. This script should handle:
1.  **Mirroring:** Copy files to local VM storage to avoid Drive I/O bottlenecks.
2.  **Sanitization:** Fix malformed CSVs (missing columns) automatically.
3.  **Archiving:** Zip results immediately to save space.
4.  **Backup:** Upload only the final ZIPs to save space.


---

## 🧩 Output Data Structure

The pipeline generates a DataFrame with the following columns:

| Column | Description |
| :--- | :--- |
| `t_code`, `ch_no`, ... | Original metadata columns. |
| `sentence` | Cleaned raw text. |
| `tokens` | List of string tokens . |
| `tokens_40` | List of token lists with len 40. |
| `pos` | List of Part-of-Speech tags (e.g., `NOUN`, `VERB`). |
| `ner` | List of Named Entity tags (e.g., `B-PER`, `O`). |
| `dep` | List of dependency tuples `(head_index, relation)`. |
| `morph` | List of morphological analysis strings. |
| `lemma` | List of root words derived from morphological analysis. |
| `sentiment` | Float (0.0 to 1.0) indicating positive sentiment probability. |

---

## 🏗 Architecture

1.  **Utility Layer (`utils_colab.py`):** Handles hardware detection (TPU/GPU), robust resource downloading with caching (`/content/vnlp_cache`), and Keras 3 compatibility helpers.
2.  **Model Layer (`vnlp_colab/[task]/`):** Contains the model architectures defined via Keras Functional API. Implements `predict_batch` methods using `@tf.function`.
3.  **Orchestration (`pipeline_colab.py`):** Manages the data flow, preprocessing, and model execution order using `tf.data.Dataset` pipelines.

---

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. 
See the `LICENSE` file for details.

Original VNLP work by [VNGRS AI](https://github.com/vngrs-ai/vnlp).
