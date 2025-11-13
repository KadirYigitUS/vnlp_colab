# tests/test_stemmer.py
import pytest
from typing import List

# Assuming the package is installed or in the python path
from vnlp_colab.stemmer.stemmer_colab import get_stemmer_analyzer, StemmerAnalyzer
from vnlp_colab.utils_colab import setup_logging

# Initialize logging once for the test module
setup_logging()

@pytest.fixture(scope="module")
def sample_tokens_stemmer() -> List[str]:
    """Provides a sample tokenized sentence for Stemmer/Morphological Analyzer tests."""
    return ["Üniversite", "sınavlarına", "canla", "başla", "çalışıyorlardı", "."]

def test_stemmer_analyzer_predict(sample_tokens_stemmer: List[str]):
    """
    Unit test for the StemmerAnalyzer model.
    Verifies output format, length, and the correctness of a few key analyses.
    """
    stemmer = get_stemmer_analyzer()
    result = stemmer.predict(sample_tokens_stemmer)

    assert isinstance(result, list)
    assert len(result) == len(sample_tokens_stemmer)
    assert all(isinstance(item, str) for item in result)

    # Validate specific, high-confidence morphological analyses
    # These represent the model's expected disambiguation choice.
    expected_analyses = {
        "Üniversite": "üniversite+Noun+A3sg+Pnon+Nom",
        "sınavlarına": "sınav+Noun+A3pl+P3sg+Dat",
        "canla": "can+Noun+A3sg+Pnon+Ins",
        "çalışıyorlardı": "çalış+Verb+Pos+Prog1+A3pl+Past",
        ".": ".+Punc",
    }

    result_dict = dict(zip(sample_tokens_stemmer, result))
    for token, expected_analysis in expected_analyses.items():
        assert result_dict.get(token) == expected_analysis

def test_stemmer_analyzer_lemmas(sample_tokens_stemmer: List[str]):
    """
    Tests the lemma extraction logic, which is a post-processing step.
    """
    stemmer = get_stemmer_analyzer()
    morph_results = stemmer.predict(sample_tokens_stemmer)
    
    # The lemma extraction is simple string splitting
    lemmas = [analysis.split('+')[0] for analysis in morph_results]

    expected_lemmas = {
        "Üniversite": "üniversite",
        "sınavlarına": "sınav",
        "çalışıyorlardı": "çalış",
        ".": ".",
    }

    result_dict = dict(zip(sample_tokens_stemmer, lemmas))
    for token, expected_lemma in expected_lemmas.items():
        assert result_dict.get(token) == expected_lemma

def test_stemmer_singleton_caching():
    """
    Tests that the get_stemmer_analyzer factory function reuses the instance.
    """
    import time
    
    # First initialization
    start_time_1 = time.time()
    stemmer1 = get_stemmer_analyzer()
    init_time_1 = time.time() - start_time_1
    
    # Second initialization should be near-instantaneous
    start_time_2 = time.time()
    stemmer2 = get_stemmer_analyzer()
    init_time_2 = time.time() - start_time_2
    
    assert stemmer1 is stemmer2
    assert init_time_2 < 0.1
    assert init_time_2 < init_time_1