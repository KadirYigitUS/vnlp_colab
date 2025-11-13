# tests/test_pipeline.py
import pytest
import pandas as pd
from pathlib import Path

# Assuming the package is installed or in the python path
from vnlp_colab.pipeline_colab import VNLPipeline

@pytest.fixture(scope="module")
def csv_fixture_path() -> Path:
    """
    Creates a temporary dummy CSV file for pipeline testing.
    This fixture has a 'module' scope, so it's created once for all tests in this file.
    """
    dummy_data = (
        "novel01\t1\t1\t1\tOnun için yol arkadaşlarımızı titizlikle seçer, kendilerini iyice sınarız.\n"
        "novel01\t1\t1\t2\tBenim adım Melikşah ve İstanbul'da yaşıyorum.\n"
        "novel01\t1\t2\t1\tBu film harikaydı, çok beğendim.\n"
    )
    # In a Colab/testing environment, /tmp is a safe place for temporary files.
    csv_path = Path("/tmp/test_input_fixture.csv")
    csv_path.write_text(dummy_data, encoding='utf-8')
    
    yield csv_path
    
    # Teardown: clean up the file after tests are done
    csv_path.unlink()
    Path("/tmp/output.initial.pkl").unlink(missing_ok=True)
    Path("/tmp/output.pkl").unlink(missing_ok=True)
    Path("/tmp/treestack_output.pkl").unlink(missing_ok=True)
    Path("/tmp/treestack_output.initial.pkl").unlink(missing_ok=True)


def test_full_spucontext_pipeline(csv_fixture_path: Path):
    """
    Integration test for the VNLPipeline with all default SPUContext models.
    Verifies that the pipeline runs end-to-end and produces a DataFrame with the correct structure.
    """
    # 1. Initialize the pipeline with all core models
    models_to_run = ['pos', 'ner', 'dep', 'stemmer', 'sentiment']
    pipeline = VNLPipeline(models_to_load=models_to_run)
    output_path = "/tmp/output.pkl"

    # 2. Run the full pipeline
    final_df = pipeline.run(csv_path=str(csv_fixture_path), output_pickle_path=output_path)

    # 3. Validate the output DataFrame
    assert isinstance(final_df, pd.DataFrame)
    assert not final_df.empty
    assert len(final_df) == 3  # Check if all rows were processed

    expected_columns = [
        't_code', 'ch_no', 'p_no', 's_no', 'sentence', 'no_accents', 'tokens',
        'tokens_40', 'sentiment', 'morph', 'lemma', 'pos', 'ner', 'dep'
    ]
    for col in expected_columns:
        assert col in final_df.columns

    # Check that the analysis columns for the first row are populated and have the correct type
    first_row = final_df.iloc[0]
    assert isinstance(first_row['sentiment'], float)
    assert isinstance(first_row['morph'], list) and first_row['morph']
    assert isinstance(first_row['lemma'], list) and first_row['lemma']
    assert isinstance(first_row['pos'], list) and first_row['pos']
    assert isinstance(first_row['ner'], list) and first_row['ner']
    assert isinstance(first_row['dep'], list) and first_row['dep']
    assert len(first_row['tokens']) == len(first_row['pos']) == len(first_row['ner']) == len(first_row['dep'])

def test_treestack_dependency_pipeline(csv_fixture_path: Path):
    """
    Integration test for the pipeline with TreeStackDP.
    This implicitly tests the dependency resolution logic (stemmer -> pos -> dep).
    """
    # 1. Initialize with a model that has a dependency chain
    models_to_run = ['dep:TreeStackDP', 'sentiment'] # Should auto-load stemmer and pos:TreeStackPoS
    pipeline = VNLPipeline(models_to_load=models_to_run)
    output_path = "/tmp/treestack_output.pkl"
    
    # 2. Run the pipeline
    final_df = pipeline.run(csv_path=str(csv_fixture_path), output_pickle_path=output_path)

    # 3. Validate the output DataFrame
    assert isinstance(final_df, pd.DataFrame)
    assert not final_df.empty

    # Check that all dependencies were run and their columns exist
    expected_columns = ['morph', 'lemma', 'pos', 'dep', 'sentiment']
    for col in expected_columns:
        assert col in final_df.columns

    # Verify content of the first row
    first_row = final_df.iloc[0]
    assert isinstance(first_row['dep'], list) and first_row['dep']
    assert len(first_row['tokens']) == len(first_row['dep'])