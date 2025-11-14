# tests/test_dep.py
import pytest
from typing import List, Tuple

# Assuming the package is installed or in the python path
from vnlp_colab.dep.dep_colab import DependencyParser
from vnlp_colab.utils_colab import setup_logging
from vnlp_colab.tokenizer_colab import TreebankWordTokenize

# Initialize logging once for the test module
setup_logging()

@pytest.fixture(scope="module")
def sample_tokens_dep() -> List[str]:
    """Provides a sample tokenized sentence for Dependency Parsing tests."""
    return TreebankWordTokenize("Onun için arkadaşlarımızı titizlikle seçeriz.")

@pytest.fixture(scope="module")
def sample_batch_tokens_dep() -> List[List[str]]:
    """Provides a batch of tokenized sentences for DP batch tests."""
    return [
        TreebankWordTokenize("Onun için arkadaşlarımızı seçeriz."),
        TreebankWordTokenize("Bu harika bir film."),
        [], # Empty sentence
        TreebankWordTokenize("Yarınki toplantı iptal edildi.")
    ]

def test_spucontext_dp_predict(sample_tokens_dep: List[str]):
    """
    Unit test for the SPUContextDP model's single predict method.
    """
    parser = DependencyParser(model='SPUContextDP')
    result = parser.predict(tokens=sample_tokens_dep)

    assert isinstance(result, list)
    assert len(result) == len(sample_tokens_dep)
    assert all(isinstance(item, tuple) and len(item) == 4 for item in result)

    expected_relations = {"seçeriz": (0, "root"), "arkadaşlarımızı": (5, "obj")}
    result_map = {item[1]: (item[2], item[3]) for item in result}
    for token, (head, label) in expected_relations.items():
        assert result_map[token][0] == head
        assert result_map[token][1] in [label, label.lower()]

def test_spucontext_dp_predict_batch(sample_batch_tokens_dep: List[List[str]]):
    """
    Unit test for the SPUContextDP model's batch prediction method.
    """
    parser = DependencyParser(model='SPUContextDP')
    batch_result = parser.predict_batch(batch_of_tokens=sample_batch_tokens_dep)
    
    assert isinstance(batch_result, list)
    assert len(batch_result) == len(sample_batch_tokens_dep)
    assert len(batch_result[2]) == 0 # Check empty sentence
    
    # Validate root predictions from different sentences
    assert batch_result[0][3][2] == 0 # "seçeriz" -> root
    assert batch_result[1][3][2] == 0 # "film" -> root
    assert batch_result[3][2][2] == 0 # "edildi" -> root

def test_treestack_dp_predict(sample_tokens_dep: List[str]):
    """
    Unit test for the TreeStackDP model's single predict method.
    """
    parser = DependencyParser(model='TreeStackDP')
    result = parser.predict(tokens=sample_tokens_dep)

    assert isinstance(result, list)
    assert len(result) == len(sample_tokens_dep)

    expected_relations = {"seçeriz": (0, "root"), "arkadaşlarımızı": (5, "obj")}
    result_map = {item[1]: (item[2], item[3]) for item in result}
    for token, (head, label) in expected_relations.items():
        assert result_map[token][0] == head
        assert result_map[token][1] in [label, label.lower()]

def test_treestack_dp_predict_batch(sample_batch_tokens_dep: List[List[str]]):
    """
    Unit test for the TreeStackDP model's batch prediction method.
    """
    parser = DependencyParser(model='TreeStackDP')
    batch_result = parser.predict_batch(batch_of_tokens=sample_batch_tokens_dep)
    
    assert isinstance(batch_result, list)
    assert len(batch_result) == len(sample_batch_tokens_dep)
    assert len(batch_result[2]) == 0
    
    # Validate root predictions from different sentences
    assert batch_result[0][3][2] == 0 # "seçeriz" -> root
    assert batch_result[1][3][2] == 0 # "film" -> root
    assert batch_result[3][2][2] == 0 # "edildi" -> root

def test_dp_singleton_caching():
    """
    Tests that the DependencyParser factory reuses instances.
    """
    import time
    
    start_time_1 = time.time()
    parser1 = DependencyParser(model='SPUContextDP')
    init_time_1 = time.time() - start_time_1
    
    start_time_2 = time.time()
    parser2 = DependencyParser(model='SPUContextDP')
    init_time_2 = time.time() - start_time_2
    
    assert parser1.instance is parser2.instance
    assert init_time_2 < 0.1
    assert init_time_2 < init_time_1