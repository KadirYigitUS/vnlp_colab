    # VNLP-Colab: A High-Performance Turkish NLP Pipeline

    This repository contains a refactored and optimized version of the VNLP library, specifically designed for high performance and compatibility with Google Colab, Keras 3, and TensorFlow 2.x+.

    ## Installation

    Install the package and all its dependencies directly from this GitHub repository into your Colab notebook:

    ```bash
    !pip install -q git+https://github.com/your-username/vnlp-colab.git
    ```

    ## Quick Start

    After installation, you can immediately import and use the main pipeline.

    ```python
    from vnlp_colab.pipeline_colab import VNLPipeline
    import pandas as pd

    # 1. Create a sample CSV file
    dummy_data = (
        "novel01\t1\t1\t1\tBu film harikaydı, çok beğendim.\n"
        "novel01\t1\t1\t2\tBenim adım Melikşah ve İstanbul'da yaşıyorum.\n"
    )
    with open("/content/input.csv", "w", encoding="utf-8") as f:
        f.write(dummy_data)

    # 2. Initialize and run the pipeline
    # Select the models you need. The pipeline handles all dependencies.
    pipeline = VNLPipeline(models_to_load=['pos', 'ner', 'sentiment', 'dep:TreeStackDP'])

    final_df = pipeline.run(
        csv_path="/content/input.csv",
        output_pickle_path="/content/analysis_results.pkl"
    )

    # 3. View the results
    print(final_df.head())
