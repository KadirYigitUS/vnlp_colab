# tests/test_dep.py
import pytest
from typing import List, Tuple

# Assuming the package is installed or in the python path
from vnlp_colab.dep.dep_colab import DependencyParser
from vnlp_colab.utils_colab import setup_logging

# Initialize logging once for the test module
setup_logging()

@pytest.fixture(scope="module")
def sample_tokens_dep() -> List[str]:
    """Provides a sample tokenized sentence for Dependency Parsing tests."""
    return ["Onun", "için", "arkadaşlarımızı", "titizlikle", "seçeriz", "."]

def test_spucontext_dp_predict(sample_tokens_dep: List[str]):
    """
    Unit test for the SPUContextDP model.
    Verifies output format, length, and key dependency relations.
    """
    parser = DependencyParser(model='SPUContextDP')
    result = parser.predict(tokens=sample_tokens_dep)

    assert isinstance(result, list)
    assert len(result) == len(sample_tokens_dep)
    
    # Check that each item is a tuple of (int, str, int, str)
    assert all(isinstance(item, tuple) and len(item) == 4 for item in result)
    assert all(
        isinstance(item[0], int) and 
        isinstance(item[1], str) and 
        isinstance(item[2], int) and 
        isinstance(item[3], str) for item in result
    )

    # Validate specific, high-confidence dependency relations
    # Format: (token, (expected_head_index, expected_label))
    expected_relations = {
        "Onun": (2, "nmod"),        # Onun -> için
        "için": (5, "obl"),         # için -> seçeriz
        "arkadaşlarımızı": (5, "obj"), # arkadaşlarımızı -> seçeriz
        "seçeriz": (0, "root"),       # seçeriz is the root
        ".": (5, "punct"),          # . -> seçeriz
    }

    result_map = {item[1]: (item[2], item[3]) for item in result}
    for token, (expected_head, expected_label) in expected_relations.items():
        assert token in result_map
        assert result_map[token][0] == expected_head
        # We can be a bit flexible with labels if they are close synonyms
        assert result_map[token][1] in [expected_label, expected_label.lower()]

def test_treestack_dp_predict(sample_tokens_dep: List[str]):
    """
    Unit test for the TreeStackDP model.
    Verifies its end-to-end functionality, including dependencies.
    """
    parser = DependencyParser(model='TreeStackDP')
    result = parser.predict(tokens=sample_tokens_dep)

    assert isinstance(result, list)
    assert len(result) == len(sample_tokens_dep)
    assert all(isinstance(item, tuple) and len(item) == 4 for item in result)

    # Validate specific, high-confidence dependency relations
    expected_relations = {
        "için": (5, "obl"),
        "arkadaşlarımızı": (5, "obj"),
        "seçeriz": (0, "root"),
        ".": (5, "punct"),
    }

    result_map = {item[1]: (item[2], item[3]) for item in result}
    for token, (expected_head, expected_label) in expected_relations.items():
        assert token in result_map
        assert result_map[token][0] == expected_head
        assert result_map[token][1] in [expected_label, expected_label.lower()]

def test_dp_singleton_caching():
    """
    Tests that the DependencyParser factory reuses instances.
    """
    import time
    
    # First initialization
    start_time_1 = time.time()
    parser1 = DependencyParser(model='SPUContextDP')
    init_time_1 = time.time() - start_time_1
    
    # Second initialization should be near-instantaneous
    start_time_2 = time.time()
    parser2 = DependencyParser(model='SPUContextDP')
    init_time_2 = time.time() - start_time_2
    
    assert parser1.instance is parser2.instance
    assert init_time_2 < 0.1
    assert init_time_2 < init_time_1