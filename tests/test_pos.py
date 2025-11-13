# tests/test_pos.py
import pytest
from typing import List, Tuple

# Assuming the package is installed or in the python path
from vnlp_colab.pos.pos_colab import PoSTagger
from vnlp_colab.utils_colab import setup_logging

# Initialize logging once for the test module
setup_logging()

@pytest.fixture(scope="module")
def sample_tokens() -> List[str]:
    """Provides a sample tokenized sentence for PoS tagging tests."""
    return ["Benim", "adım", "Melikşah", "ve", "İstanbul'da", "yaşıyorum", "."]

def test_spucontext_pos_tagger_predict(sample_tokens: List[str]):
    """
    Unit test for the SPUContextPoS model.
    Verifies output format, length, and key tag correctness.
    """
    pos_tagger = PoSTagger(model='SPUContextPoS')
    result = pos_tagger.predict(sample_tokens)

    assert isinstance(result, list)
    assert len(result) == len(sample_tokens)
    
    # Check that each item is a tuple of (string, string)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
    assert all(isinstance(item[0], str) and isinstance(item[1], str) for item in result)

    # Validate specific, high-confidence tags
    expected_tags = {
        "Melikşah": "PROPN",
        "ve": "CCONJ",
        "İstanbul'da": "PROPN",
        "yaşıyorum": "VERB",
        ".": "PUNCT",
    }

    result_dict = dict(result)
    for token, expected_tag in expected_tags.items():
        assert result_dict.get(token) == expected_tag

def test_treestack_pos_tagger_predict(sample_tokens: List[str]):
    """
    Unit test for the TreeStackPoS model.
    Verifies its end-to-end functionality, including its dependency on the stemmer.
    """
    pos_tagger = PoSTagger(model='TreeStackPoS')
    result = pos_tagger.predict(sample_tokens)

    assert isinstance(result, list)
    assert len(result) == len(sample_tokens)
    
    assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    # Validate specific, high-confidence tags
    expected_tags = {
        "Melikşah": "PROPN",
        "ve": "CCONJ",
        "İstanbul'da": "PROPN",
        "yaşıyorum": "VERB",
        ".": "PUNCT",
    }

    result_dict = dict(result)
    for token, expected_tag in expected_tags.items():
        assert result_dict.get(token) == expected_tag

def test_pos_tagger_singleton_caching():
    """
    Tests that the PoSTagger factory reuses instances instead of re-creating them.
    """
    import time
    
    # First initialization should be slow (downloads/loads model)
    start_time_1 = time.time()
    tagger1 = PoSTagger(model='SPUContextPoS')
    init_time_1 = time.time() - start_time_1
    
    # Second initialization should be near-instantaneous
    start_time_2 = time.time()
    tagger2 = PoSTagger(model='SPUContextPoS')
    init_time_2 = time.time() - start_time_2
    
    assert tagger1.instance is tagger2.instance
    assert init_time_2 < 0.1  # Subsequent calls should be extremely fast
    assert init_time_2 < init_time_1