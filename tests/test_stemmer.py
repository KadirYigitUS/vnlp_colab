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

@pytest.fixture(scope="module")
def sample_tokens_batch_stemmer() -> List[List[str]]:
    """Provides a batch of tokenized sentences for testing batch prediction."""
    return [
        ["Benim", "adım", "Melikşah", "."],
        ["Vapurla", "Beşiktaş'a", "geçip", "ulaştım", "."],
        [], # Test case with an empty sentence
        ["Bu", "sadece", "bir", "test", "."]
    ]


def test_stemmer_analyzer_predict(sample_tokens_stemmer: List[str]):
    """
    Unit test for the StemmerAnalyzer model's single predict method.
    Verifies output format, length, and the correctness of a few key analyses.
    """
    stemmer = get_stemmer_analyzer()
    result = stemmer.predict(sample_tokens_stemmer)

    assert isinstance(result, list)
    assert len(result) == len(sample_tokens_stemmer)
    assert all(isinstance(item, str) for item in result)

    # Validate specific, high-confidence morphological analyses
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

def test_stemmer_analyzer_predict_batch(sample_tokens_batch_stemmer: List[List[str]]):
    """
    Unit test for the StemmerAnalyzer's batch prediction method.
    Verifies output structure and correctness for a multi-sentence batch.
    """
    stemmer = get_stemmer_analyzer()
    batch_result = stemmer.predict_batch(sample_tokens_batch_stemmer)

    assert isinstance(batch_result, list)
    assert len(batch_result) == len(sample_tokens_batch_stemmer)
    
    # Check dimensions and types
    assert len(batch_result[0]) == 4
    assert len(batch_result[1]) == 5
    assert len(batch_result[2]) == 0
    assert len(batch_result[3]) == 5
    assert isinstance(batch_result[0][0], str)

    # Validate specific analyses from different sentences in the batch
    assert batch_result[0][2] == "melikşah+Noun+Prop+A3sg+Pnon+Nom"
    assert batch_result[1][1] == "beşiktaş+Noun+Prop+A3sg+Pnon+Dat"
    assert batch_result[3][3] == "test+Noun+A3sg+Pnon+Nom"


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