# tests/test_ner.py
import pytest
from typing import List, Tuple

# Assuming the package is installed or in the python path
from vnlp_colab.ner.ner_colab import NamedEntityRecognizer
from vnlp_colab.utils_colab import setup_logging
from vnlp_colab.tokenizer_colab import WordPunctTokenize, TreebankWordTokenize


# Initialize logging once for the test module
setup_logging()

@pytest.fixture(scope="module")
def sample_sentence_ner() -> str:
    """Provides a sample sentence for NER tests."""
    return "Benim adım Melikşah ve VNGRS AI Takımı'nda çalışıyorum."

@pytest.fixture(scope="module")
def sample_tokens_ner() -> List[str]:
    """Provides the corresponding tokens for the sample sentence."""
    return WordPunctTokenize("Benim adım Melikşah ve VNGRS AI Takımı'nda çalışıyorum.")

@pytest.fixture(scope="module")
def sample_batch_ner() -> Tuple[List[str], List[List[str]]]:
    """Provides a batch of sentences and their tokens for NER tests."""
    sentences = [
        "Ali Bey, Ankara'ya gitti.",
        "Toplantı VNGRS ofisinde yapılacak.",
        "Ayşe Hanım nerede?"
    ]
    tokens = [WordPunctTokenize(s) for s in sentences]
    return sentences, tokens


def test_spucontext_ner_predict(sample_sentence_ner: str, sample_tokens_ner: List[str]):
    """
    Unit test for the SPUContextNER model's single predict method.
    """
    ner = NamedEntityRecognizer(model='SPUContextNER')
    result = ner.predict(sentence=sample_sentence_ner, tokens=sample_tokens_ner)

    assert isinstance(result, list)
    assert len(result) == len(sample_tokens_ner)
    assert all(isinstance(item, tuple) for item in result)

    expected_tags = {"Melikşah": "PER", "VNGRS": "ORG", "AI": "ORG"}
    result_dict = dict(result)
    for token, expected_tag in expected_tags.items():
        assert result_dict.get(token) == expected_tag

def test_spucontext_ner_predict_batch(sample_batch_ner: Tuple[List[str], List[List[str]]]):
    """
    Unit test for the SPUContextNER model's batch prediction method.
    """
    ner = NamedEntityRecognizer(model='SPUContextNER')
    sentences, tokens = sample_batch_ner
    batch_result = ner.predict_batch(batch_of_sentences=sentences, batch_of_tokens=tokens)

    assert isinstance(batch_result, list)
    assert len(batch_result) == len(sentences)
    
    # Check specific tags from different sentences
    result_dict_0 = dict(batch_result[0])
    result_dict_1 = dict(batch_result[1])
    result_dict_2 = dict(batch_result[2])
    
    assert result_dict_0.get("Ali") == "PER"
    assert result_dict_0.get("Ankara'ya") == "LOC"
    assert result_dict_1.get("VNGRS") == "ORG"
    assert result_dict_2.get("Ayşe") == "PER"

def test_charner_predict():
    """
    Unit test for the CharNER model's single predict method.
    """
    ner = NamedEntityRecognizer(model='CharNER')
    sentence = "VNGRS AI Takımı'nda çalışıyorum."
    result = ner.predict(sentence=sentence, tokens=[]) # Tokens are ignored

    assert isinstance(result, list)
    result_dict = dict(result)
    expected_tags = {"VNGRS": "ORG", "AI": "ORG", "Takımı": "ORG"}
    for token, expected_tag in expected_tags.items():
        assert result_dict.get(token) == expected_tag

def test_charner_predict_batch(sample_batch_ner: Tuple[List[str], List[List[str]]]):
    """
    Unit test for the CharNER model's batch prediction method.
    """
    ner = NamedEntityRecognizer(model='CharNER')
    sentences, _ = sample_batch_ner # Tokens are ignored by CharNER
    
    # We pass empty token lists for API consistency
    empty_tokens = [[] for _ in sentences]
    batch_result = ner.predict_batch(batch_of_sentences=sentences, batch_of_tokens=empty_tokens)
    
    assert isinstance(batch_result, list)
    assert len(batch_result) == len(sentences)
    
    # Check specific tags from different sentences
    result_dict_0 = dict(batch_result[0])
    result_dict_1 = dict(batch_result[1])
    result_dict_2 = dict(batch_result[2])
    
    assert result_dict_0.get("Ali") == "PER"
    assert result_dict_0.get("Ankara") == "LOC" # CharNER tokenizes differently
    assert result_dict_1.get("VNGRS") == "ORG"
    assert result_dict_2.get("Ayşe") == "PER"

def test_ner_singleton_caching():
    """
    Tests that the NamedEntityRecognizer factory reuses instances.
    """
    import time
    
    start_time_1 = time.time()
    ner1 = NamedEntityRecognizer(model='SPUContextNER')
    init_time_1 = time.time() - start_time_1
    
    start_time_2 = time.time()
    ner2 = NamedEntityRecognizer(model='SPUContextNER')
    init_time_2 = time.time() - start_time_2
    
    assert ner1.instance is ner2.instance
    assert init_time_2 < 0.1
    assert init_time_2 < init_time_1