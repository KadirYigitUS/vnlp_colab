# tests/test_dep.py
"""Unit tests for Dependency Parser, chunking utilities, and merge invariants."""
from typing import List
import pytest

from tests._dep_fakes import DeterministicFakeDP, make_synthetic_tokens
from vnlp_colab.dep.dep_colab import DependencyParser
from vnlp_colab.dep.dep_utils_colab import chunk_token_sequence, merge_dep_chunks
from vnlp_colab.tokenizer_colab import TreebankWordTokenize

try:
    import keras
    keras_version = int(getattr(keras, "__version__", "2").split(".")[0])
except Exception:
    keras_version = 2

skip_if_not_keras3 = pytest.mark.skipif(
    keras_version < 3, reason="Live neural model inference requires Keras 3 runtime"
)


def test_chunk_token_sequence():
    """Verify contiguous splitting, boundary preservation, and invalid chunk sizes."""
    assert chunk_token_sequence([]) == []
    t40 = make_synthetic_tokens(40)
    c40 = chunk_token_sequence(t40, 40)
    assert len(c40) == 1 and len(c40[0]) == 40
    c44 = chunk_token_sequence(make_synthetic_tokens(44), 40)
    assert len(c44) == 2 and len(c44[0]) == 40 and len(c44[1]) == 4
    c85 = chunk_token_sequence(make_synthetic_tokens(85), 40)
    assert len(c85) == 3 and [len(c) for c in c85] == [40, 40, 5]
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_token_sequence(t40, 0)


def test_merge_dep_chunks_invariants_and_clamping():
    """Verify offset shifting, OOB clamping, trailing chunk trimming, and YG-ZB tail."""
    assert merge_dep_chunks([]) == []
    c1 = [(1, "A", 0, "root"), (2, "B", 1, "nmod")]
    assert merge_dep_chunks([c1], [["A", "B"]]) == c1

    # YG-ZB (1,4,17) tail simulation: 40 tokens + 4-token tail
    chunk0 = [(i + 1, f"t{i+1}", 0 if i == 0 else 1, "root" if i == 0 else "dep") for i in range(40)]
    chunk1 = [(1, "yeni", 2, "amod"), (2, "sözcük", 3, "obj"), (3, "türetildi", 0, "root"), (4, ".", 3, "punct")]
    merged_yg_zb = merge_dep_chunks([chunk0, chunk1])
    assert len(merged_yg_zb) == 44
    assert merged_yg_zb[40] == (41, "yeni", 42, "amod")
    assert merged_yg_zb[41] == (42, "sözcük", 43, "obj")
    assert merged_yg_zb[42] == (43, "türetildi", 0, "root")
    assert merged_yg_zb[43] == (44, ".", 43, "punct")

    # KT-OK (8,136,3869) OOB simulation: chunk_len=23, local_head=25 > 23
    oob_telemetry = {}
    chunk_oob = [(1, "w1", 25, "dep"), (2, "w2", 0, "root")]
    merged_oob = merge_dep_chunks([chunk0, chunk_oob], oob_telemetry=oob_telemetry)
    assert oob_telemetry.get("dep_head_oob") == 1
    # Clamped to chunk_len 2 -> local 2 -> shifted by +40 = 42
    assert merged_oob[40] == (41, "w1", 42, "dep")

    # Trailing empty chunk guard
    merged_trimmed = merge_dep_chunks([chunk0, []])
    assert len(merged_trimmed) == 40


@pytest.mark.parametrize("model_name", ["SPUContextDP", "TreeStackDP"])
@pytest.mark.parametrize("length", [40, 44, 85])
def test_deterministic_fake_dp_multi_chunk(model_name: str, length: int):
    """Test fake parser preserving token IDs 1..N, root 0, and shifted heads in k>=1."""
    parser = DeterministicFakeDP(model_name=model_name)
    tokens = make_synthetic_tokens(length)
    single_res = parser.predict(tokens)
    batch_res = parser.predict_batch([tokens, []])
    assert single_res == batch_res[0]
    assert batch_res[1] == []
    assert len(single_res) == length
    for idx, (tok_id, tok, head, rel) in enumerate(single_res):
        assert tok_id == idx + 1
        k = idx // 40
        if idx % 40 == 0:
            assert head == 0 and rel == "root"
        else:
            assert head > 0 and 40 * k + 1 <= head <= 40 * k + 40


@pytest.fixture(scope="module")
def sample_tokens_dep() -> List[str]:
    return TreebankWordTokenize("Onun için arkadaşlarımızı titizlikle seçeriz.")


@pytest.fixture(scope="module")
def sample_batch_tokens_dep() -> List[List[str]]:
    return [
        TreebankWordTokenize("Onun için arkadaşlarımızı seçeriz."),
        TreebankWordTokenize("Bu harika bir film."),
        [],
        TreebankWordTokenize("Yarınki toplantı iptal edildi."),
    ]


@skip_if_not_keras3
def test_spucontext_dp_predict(sample_tokens_dep: List[str]):
    parser = DependencyParser(model="SPUContextDP")
    result = parser.predict(tokens=sample_tokens_dep)
    assert len(result) == len(sample_tokens_dep)
    result_map = {item[1]: (item[2], item[3]) for item in result}
    assert result_map["seçeriz"][0] == 0


@skip_if_not_keras3
def test_spucontext_dp_predict_batch(sample_batch_tokens_dep: List[List[str]]):
    parser = DependencyParser(model="SPUContextDP")
    batch_result = parser.predict_batch(batch_of_tokens=sample_batch_tokens_dep)
    assert len(batch_result) == len(sample_batch_tokens_dep)
    assert batch_result[0][3][2] == 0


@skip_if_not_keras3
def test_treestack_dp_predict(sample_tokens_dep: List[str]):
    parser = DependencyParser(model="TreeStackDP")
    result = parser.predict(tokens=sample_tokens_dep)
    assert len(result) == len(sample_tokens_dep)


@skip_if_not_keras3
def test_treestack_dp_predict_batch(sample_batch_tokens_dep: List[List[str]]):
    parser = DependencyParser(model="TreeStackDP")
    batch_result = parser.predict_batch(batch_of_tokens=sample_batch_tokens_dep)
    assert len(batch_result) == len(sample_batch_tokens_dep)


@skip_if_not_keras3
def test_dp_singleton_caching():
    parser1 = DependencyParser(model="SPUContextDP")
    parser2 = DependencyParser(model="SPUContextDP")
    assert parser1.instance is parser2.instance
