# tests/test_pos.py
import pytest
from typing import List, Tuple

# Assuming the package is installed or in the python path
from vnlp_colab.pos.pos_colab import PoSTagger
from vnlp_colab.utils_colab import setup_logging
from vnlp_colab.tokenizer_colab import TreebankWordTokenize

# Initialize logging once for the test module
setup_logging()

@pytest.fixture(scope="module")
def sample_tokens() -> List[str]:
    """Provides a sample tokenized sentence for PoS tagging tests."""
    return ["Benim", "adım", "Melikşah", "ve", "İstanbul'da", "yaşıyorum", "."]

@pytest.fixture(scope="module")
def sample_batch_tokens() -> List[List[str]]:
    """Provides a batch of tokenized sentences for testing batch prediction."""
    return [
        TreebankWordTokenize("Benim adım Melikşah."),
        TreebankWordTokenize("O, İstanbul'da yaşıyor."),
        [],
        TreebankWordTokenize("Bu bir test cümlesidir.")
    ]

def test_spucontext_pos_tagger_predict(sample_tokens: List[str]):
    """
    Unit test for the SPUContextPoS model's single predict method.
    Verifies output format, length, and key tag correctness.
    """
    pos_tagger = PoSTagger(model='SPUContextPoS')
    result = pos_tagger.predict(sample_tokens)

    assert isinstance(result, list)
    assert len(result) == len(sample_tokens)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    expected_tags = {"Melikşah": "PROPN", "ve": "CCONJ", "İstanbul'da": "PROPN", "yaşıyorum": "VERB"}
    result_dict = dict(result)
    for token, expected_tag in expected_tags.items():
        assert result_dict.get(token) == expected_tag

def test_spucontext_pos_tagger_predict_batch(sample_batch_tokens: List[List[str]]):
    """
    Unit test for the SPUContextPoS model's batch prediction method.
    """
    pos_tagger = PoSTagger(model='SPUContextPoS')
    batch_result = pos_tagger.predict_batch(sample_batch_tokens)

    assert isinstance(batch_result, list)
    assert len(batch_result) == len(sample_batch_tokens)
    
    # Check dimensions and types
    assert len(batch_result[0]) == 4
    assert len(batch_result[1]) == 5
    assert len(batch_result[2]) == 0
    assert isinstance(batch_result[0][0], tuple)

    # Validate specific tags from different sentences in the batch
    result_dict_0 = dict(batch_result[0])
    result_dict_1 = dict(batch_result[1])
    assert result_dict_0.get("Melikşah") == "PROPN"
    assert result_dict_1.get("İstanbul'da") == "PROPN"
    assert result_dict_1.get("yaşıyor") == "VERB"


def test_treestack_pos_tagger_predict(sample_tokens: List[str]):
    """
    Unit test for the TreeStackPoS model's single predict method.
    """
    pos_tagger = PoSTagger(model='TreeStackPoS')
    result = pos_tagger.predict(sample_tokens)

    assert isinstance(result, list)
    assert len(result) == len(sample_tokens)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    expected_tags = {"Melikşah": "PROPN", "ve": "CCONJ", "İstanbul'da": "PROPN", "yaşıyorum": "VERB"}
    result_dict = dict(result)
    for token, expected_tag in expected_tags.items():
        assert result_dict.get(token) == expected_tag

def test_treestack_pos_tagger_predict_batch(sample_batch_tokens: List[List[str]]):
    """
    Unit test for the TreeStackPoS model's batch prediction method.
    """
    pos_tagger = PoSTagger(model='TreeStackPoS')
    batch_result = pos_tagger.predict_batch(sample_batch_tokens)

    assert isinstance(batch_result, list)
    assert len(batch_result) == len(sample_batch_tokens)

    # Validate specific tags from different sentences in the batch
    result_dict_0 = dict(batch_result[0])
    result_dict_1 = dict(batch_result[1])
    assert result_dict_0.get("Melikşah") == "PROPN"
    assert result_dict_1.get("İstanbul'da") == "PROPN"


def test_pos_tagger_singleton_caching():
    """
    Tests that the PoSTagger factory reuses instances instead of re-creating them.
    """
    import time
    
    start_time_1 = time.time()
    tagger1 = PoSTagger(model='SPUContextPoS')
    init_time_1 = time.time() - start_time_1
    
    start_time_2 = time.time()
    tagger2 = PoSTagger(model='SPUContextPoS')
    init_time_2 = time.time() - start_time_2
    
    assert tagger1.instance is tagger2.instance
    assert init_time_2 < 0.1
    assert init_time_2 < init_time_1