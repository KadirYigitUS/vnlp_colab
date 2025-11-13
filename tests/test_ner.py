# tests/test_ner.py
import pytest
from typing import List, Tuple

# Assuming the package is installed or in the python path
from vnlp_colab.ner.ner_colab import NamedEntityRecognizer
from vnlp_colab.utils_colab import setup_logging

# Initialize logging once for the test module
setup_logging()

@pytest.fixture(scope="module")
def sample_sentence_ner() -> str:
    """Provides a sample sentence for NER tests."""
    return "Benim adım Melikşah ve VNGRS AI Takımı'nda çalışıyorum."

@pytest.fixture(scope="module")
def sample_tokens_ner() -> List[str]:
    """Provides the corresponding tokens for the sample sentence."""
    return ["Benim", "adım", "Melikşah", "ve", "VNGRS", "AI", "Takımı'nda", "çalışıyorum", "."]

def test_spucontext_ner_predict(sample_sentence_ner: str, sample_tokens_ner: List[str]):
    """
    Unit test for the SPUContextNER model.
    Verifies output format, length, and key tag correctness.
    """
    ner = NamedEntityRecognizer(model='SPUContextNER')
    result = ner.predict(sentence=sample_sentence_ner, tokens=sample_tokens_ner)

    assert isinstance(result, list)
    assert len(result) == len(sample_tokens_ner)
    
    # Check that each item is a tuple of (string, string)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
    assert all(isinstance(item[0], str) and isinstance(item[1], str) for item in result)

    # Validate specific, high-confidence tags
    expected_tags = {
        "Melikşah": "PER",
        "VNGRS": "ORG",
        "AI": "ORG",
        "Takımı'nda": "ORG",
        "adım": "O",
    }

    result_dict = dict(result)
    for token, expected_tag in expected_tags.items():
        assert result_dict.get(token) == expected_tag

def test_charner_predict():
    """
    Unit test for the CharNER model.
    Note: CharNER does its own internal tokenization (WordPunct), so we test
    it with a raw sentence and check against its expected tokenization.
    """
    ner = NamedEntityRecognizer(model='CharNER')
    sentence = "VNGRS AI Takımı'nda çalışıyorum."
    result = ner.predict(sentence=sentence, tokens=[]) # Tokens are ignored by CharNER

    assert isinstance(result, list)
    
    result_dict = dict(result)
    
    # CharNER uses WordPunctTokenize, which splits "'nda"
    expected_tags = {
        "VNGRS": "ORG",
        "AI": "ORG",
        "Takımı": "ORG",
        "'": "ORG", # Continues the ORG entity
        "nda": "ORG",
        "çalışıyorum": "O",
    }
    
    for token, expected_tag in expected_tags.items():
        assert result_dict.get(token) == expected_tag

def test_ner_singleton_caching():
    """
    Tests that the NamedEntityRecognizer factory reuses instances.
    """
    import time
    
    # First initialization
    start_time_1 = time.time()
    ner1 = NamedEntityRecognizer(model='SPUContextNER')
    init_time_1 = time.time() - start_time_1
    
    # Second initialization should be near-instantaneous
    start_time_2 = time.time()
    ner2 = NamedEntityRecognizer(model='SPUContextNER')
    init_time_2 = time.time() - start_time_2
    
    assert ner1.instance is ner2.instance
    assert init_time_2 < 0.1
    assert init_time_2 < init_time_1