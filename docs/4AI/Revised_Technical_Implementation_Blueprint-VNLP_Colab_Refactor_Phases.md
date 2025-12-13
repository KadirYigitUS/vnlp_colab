## Revised Technical Implementation Blueprint: VNLP Colab Refactor & Deployment

This blueprint outlines the comprehensive plan to refactor the VNLP package for Google Colab, ensuring compatibility with Keras 3 and modern TensorFlow. It also serves as a step-by-step guide covering code generation, a detailed Colab testing workflow for a non-GPU local environment, Python packaging, and version control with GitHub.

### Phase 1: Code Generation & Local Setup (AI-Driven)

This phase focuses on the AI's task of generating the refactored code and necessary configuration files. As the user, your role here is to save these generated files to a local project directory.

**1.1. Project Structure:**

First, create the following directory structure on your local machine. This structure is crucial for the subsequent Colab, packaging, and GitHub steps.

```
vnlp_colab/
├── vnlp_colab/
│   ├── __init__.py
│   └── utils_colab.py
├── tests/
│   └── test_vnlp_colab.py
├── pyproject.toml
├── setup.cfg
└── .gitignore
```

**1.2. Core Module Generation: `vnlp_colab/utils_colab.py`**

I will generate a fully refactored `utils_colab.py` with the following enhancements:

*   **Modern Python & Readability:**
    *   **Header:** Comprehensive module-level docstring, `VERSION` constant, and organized imports.
    *   **Logging:** A `logging.basicConfig` setup will replace all `print()` statements for structured, level-based output (INFO, WARNING, ERROR).
    *   **Path Management:** All file paths will be handled using `pathlib.Path` for OS-agnostic compatibility.
*   **Colab & Keras 3 Compatibility:**
    *   **Hardware Detection:** A `get_strategy()` function will automatically detect and return a `tf.distribute.Strategy` for GPU or TPU, ensuring models utilize available accelerators.
    *   **Networking & Caching:** The `check_and_download` function will be modernized to:
        *   Use `requests` with streaming and a `tqdm` progress bar for a superior download experience.
        *   Save all assets to a centralized cache directory (`/content/vnlp_cache` in Colab) to avoid re-downloads.
        *   Validate file integrity with optional hash checking.
    *   **Keras 3 API:** All model creation helpers (e.g., `create_rnn_stacks`) and tokenizer loading functions will be updated to be fully compliant with Keras 3 and the latest TensorFlow version in Colab.

**1.3. Model Architecture Refactoring Strategy:**

The core logic within the `vnlp` subpackages (POS, DEP, NER, Stemmer) will be updated as follows:

*   **Functional API Conversion:** Model definitions like `create_spucontext_ner_model` will be converted from the legacy `Sequential` API to the modern Keras **Functional API**. This makes the model architecture explicit, resolves Keras 3 compatibility issues, and is essential for loading the provided weights correctly. The architecture blueprints you provided will be the definitive guide.
*   **Performance with `@tf.function`:** The core prediction logic inside each model's `predict` method will be wrapped in a `@tf.function`. This compiles the Python code into a high-performance TensorFlow graph, drastically speeding up the autoregressive prediction loops in Colab.
*   **Singleton Pattern Enhancement:** The singleton factories (`get_ner_instance`, `get_tagger_instance`, etc.) will be reviewed to ensure they are thread-safe and efficiently cache model instances, preventing slow re-initialization in notebooks.

**1.4. Testing Script Generation: `tests/test_vnlp_colab.py`**

I will generate a `pytest`-compatible test file containing:
*   Unit tests for the new utility functions in `utils_colab.py` (e.g., testing the downloader and path logic).
*   An integration test that loads a complete model (e.g., `PoSTagger`) and runs a prediction on a sample sentence to verify end-to-end functionality.

**1.5. Packaging File Generation: `pyproject.toml` & `setup.cfg`**

I will provide the content for these two files, which are necessary to build a distributable Python package. They will define the project name, version, dependencies, and package structure.

### Phase 2: Colab Environment Setup & Testing Workflow (User-Driven)

This is your step-by-step guide to testing the refactored code on Google Colab.

**Step 2.1: Prepare Colab Notebook**
1.  Open a new Colab notebook at [colab.research.google.com](https://colab.research.google.com).
2.  Go to **Runtime -> Change runtime type**.
3.  Select **T4 GPU** from the "Hardware accelerator" dropdown and click **Save**. This provides a free GPU for testing.

**Step 2.2: Sync Project Files via Google Drive**
1.  On your local machine, upload the entire `vnlp_colab/` project folder to the root of your Google Drive.
2.  In your Colab notebook, run the following cell to mount your Google Drive. You will be prompted to authorize access.
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
3.  Verify that your project is accessible by listing its contents:
    ```bash
    !ls /content/drive/MyDrive/vnlp_colab/
    ```
    You should see your `vnlp_colab/` directory, `tests/`, `pyproject.toml`, etc.

**Step 2.3: Install Dependencies**
Run the following cell to install the necessary libraries. The `-q` flag ensures a quiet installation.
```bash
%pip install -q keras==3.4.1 tensorflow sentencepiece==0.2.0 regex requests==2.32.3 spylls tqdm pytest
```
*Note: We pin `keras` to a known stable version. TensorFlow will be automatically handled by Colab.*

**Step 2.4: Run Automated Unit Tests**
This step verifies that the core components are working as expected.
```bash
!python -m pytest /content/drive/MyDrive/vnlp_colab/tests/test_vnlp_colab.py
```
Look for a "passed" status at the end of the output. This confirms the low-level functions are correct.

**Step 2.5: Perform Interactive Integration Testing**
This is the final validation step. You will import and use one of the main VNLP classes directly in the notebook.

1.  Add your project directory to Python's path:
    ```python
    import sys
    sys.path.append('/content/drive/MyDrive/vnlp_colab')
    ```
2.  Import a model, instantiate it, and run a prediction. The first time you run this, it will download the model weights and cache them, showing a `tqdm` progress bar.
    ```python
    from vnlp_colab.part_of_speech_tagger import PoSTagger # Assuming PoSTagger is refactored

    # First initialization will be slow and show download progress
    print("Initializing PoSTagger for the first time...")
    pos_tagger = PoSTagger(model='SPUContextPoS')

    # Subsequent initializations will be instant (demonstrates singleton caching)
    print("\nRe-initializing PoSTagger (should be instant)...")
    pos_tagger_2 = PoSTagger(model='SPUContextPoS')

    # Run prediction
    sentence = "Bu, Colab üzerinde çalışan modernize edilmiş bir test cümlesidir."
    print(f"\nPredicting POS tags for: '{sentence}'")
    pos_result = pos_tagger.predict(sentence)
    print(pos_result)
    ```

### Phase 3: Packaging for Distribution (User-Driven)

Once testing is complete, you can build the project into a distributable package locally.

**Step 3.1: Install Build Tools**
Open a terminal on your local Ubuntu machine and run:
```bash
pip install build
```

**Step 3.2: Build the Package**
Navigate to the root of your `vnlp_colab/` project directory (where `pyproject.toml` is) and run:
```bash
python -m build
```
This will create a `dist/` folder containing a `.tar.gz` (source archive) and a `.whl` (wheel) file. These are your distributable package files.

### Phase 4: Version Control and GitHub Workflow (User-Driven)

Finally, upload your completed, tested, and packaged project to GitHub.

**Step 4.1: Initialize Git Repository**
In your local `vnlp_colab/` project root, run:
```bash
git init
git add .
git commit -m "Initial refactor for Colab and Keras 3 compatibility"
```

**Step 4.2: Create GitHub Repository**
1.  Go to [GitHub](https://github.com) and create a new repository (e.g., `vnlp-colab`). Do **not** initialize it with a README or license file.
2.  GitHub will provide you with commands for an existing repository. Copy the URL.

**Step 4.3: Push to GitHub**
Back in your local terminal, run the following commands, replacing the URL with your own:
```bash
git remote add origin https://github.com/your-username/vnlp-colab.git
git branch -M main
git push -u origin main
```
