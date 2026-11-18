# tests/test_pipeline.py
"""Integration tests for VNLPipeline, dependency chunking, and idempotent migration."""
from pathlib import Path
import pandas as pd
import pytest

from tests._dep_fakes import DeterministicFakeDP, make_synthetic_tokens
from vnlp_colab.dep.dep_migration import adjust_dataframe_dep_chunks
from vnlp_colab.dep.dep_utils_colab import chunk_token_sequence
from vnlp_colab.pipeline_colab import VNLPipeline

try:
    import keras
    keras_version = int(getattr(keras, "__version__", "2").split(".")[0])
except Exception:
    keras_version = 2

skip_if_not_keras3 = pytest.mark.skipif(
    keras_version < 3, reason="Live neural pipeline requires Keras 3 runtime"
)


@pytest.fixture(scope="module")
def csv_fixture_path() -> Path:
    dummy_data = (
        "novel01\t1\t1\t1\tOnun için yol arkadaşlarımızı titizlikle seçer, kendilerini iyice sınarız.\n"
        "novel01\t1\t1\t2\tTest.\n"
        "novel01\t1\t1\t3\tBenim adım Melikşah ve İstanbul'da yaşıyorum.\n"
        "novel01\t1\t2\t1\tBu film harikaydı, çok beğendim.\n"
        "novel01\t1\t2\t2\t\n"
    )
    csv_path = Path("/tmp/test_input_fixture.csv")
    csv_path.write_text(dummy_data, encoding="utf-8")
    yield csv_path
    csv_path.unlink(missing_ok=True)
    for p in ["/tmp/output.initial.pkl", "/tmp/output.pkl", "/tmp/treestack_output.pkl", "/tmp/treestack_output.initial.pkl"]:
        Path(p).unlink(missing_ok=True)


@skip_if_not_keras3
def test_full_spucontext_pipeline(csv_fixture_path: Path):
    pipeline = VNLPipeline(models_to_load=["pos", "ner", "dep", "stemmer", "sentiment"])
    final_df = pipeline.run(csv_path=str(csv_fixture_path), output_pickle_path="/tmp/output.pkl", batch_size=2)
    assert isinstance(final_df, pd.DataFrame) and len(final_df) == 4


@skip_if_not_keras3
def test_treestack_dependency_pipeline(csv_fixture_path: Path):
    pipeline = VNLPipeline(models_to_load=["dep:TreeStackDP", "sentiment"])
    final_df = pipeline.run(csv_path=str(csv_fixture_path), output_pickle_path="/tmp/treestack_output.pkl", batch_size=2)
    assert isinstance(final_df, pd.DataFrame) and len(final_df) == 4


def test_pipeline_long_sentences_no_double_offset():
    """Verify binary (head, rel) from process_dataframe match DependencyParser.predict with 0 double offset."""
    pipeline = VNLPipeline(models_to_load=[])
    fake_dp = DeterministicFakeDP()
    pipeline.models = {"dep": fake_dp}

    lengths = [40, 44, 85]
    tokens_list = [make_synthetic_tokens(ln) for ln in lengths]
    df = pd.DataFrame({
        "sentence": [f"sent_{ln}" for ln in lengths],
        "no_accents": [f"sent_{ln}" for ln in lengths],
        "tokens": tokens_list,
        "tokens_40": [chunk_token_sequence(t, 40) for t in tokens_list],
    })

    processed_df = pipeline.process_dataframe(df, batch_size=2)
    assert "dep" in processed_df.columns
    for idx, ln in enumerate(lengths):
        dep_res = processed_df.loc[idx, "dep"]
        assert len(dep_res) == ln
        expected = [(item[2], item[3]) for item in fake_dp.predict(tokens_list[idx])]
        assert dep_res == expected
        if ln > 40:
            # Chunk 1 (idx 40..ln-1): head must be shifted by +40, not double shifted (+80)
            chunk1_heads = [h for (h, _) in dep_res[40:min(80, ln)]]
            assert all(h == 0 or 41 <= h <= 80 for h in chunk1_heads)
            assert not any(h > 80 for h in chunk1_heads)


def test_adjust_dataframe_dep_chunks_idempotent():
    """Verify YG-ZB tail, KT-OK OOB clamping, 3-chunk, list/str parsing, English guard, and Pass-2 0-diff."""
    # 1. YG-ZB (1,4,17): 40 tokens + 4-token tail with local heads
    t40_yg = [make_synthetic_tokens(40), ["yeni", "sözcük", "türetildi", "."]]
    dep_yg = [(1, "root")] + [(1, "dep")] * 39 + [(2, "amod"), (3, "obj"), (0, "root"), (3, "punct")]

    # 2. KT-OK (8,136,3869): chunk_len=23, local_head=25 > 23 (OOB)
    t40_kt = [make_synthetic_tokens(40), make_synthetic_tokens(23)]
    dep_kt = [(1, "root")] + [(1, "dep")] * 39 + [(25, "dep")] + [(0, "root")] * 22

    # 3. 3-chunk native len=85
    t40_85 = [make_synthetic_tokens(40), make_synthetic_tokens(40), make_synthetic_tokens(5)]
    dep_85 = [(1, "root")] + [(1, "dep")] * 39 + [(1, "root")] + [(1, "dep")] * 39 + [(1, "root")] + [(1, "dep")] * 4

    df = pd.DataFrame({
        "tokens_40": [t40_yg, repr(t40_kt), t40_85, None],
        "dep": [repr(dep_yg), dep_kt, dep_85, [(1, "root")] * 10],
    })

    # Pass 1: Apply migration
    df_migrated, telem1 = adjust_dataframe_dep_chunks(df, chunk_size=40)
    assert telem1["rows_modified"] == 3
    assert telem1["dep_head_oob"] == 1  # KT-OK OOB clamped

    # Inspect YG-ZB shifted heads in migrated DF
    yg_migrated_dep = eval(df_migrated.at[0, "dep"])
    assert yg_migrated_dep[40:] == [(42, "amod"), (43, "obj"), (0, "root"), (43, "punct")]

    # Inspect KT-OK clamped & shifted head (local 25 -> clamped 23 -> shifted 63)
    kt_migrated_dep = df_migrated.at[1, "dep"]
    assert kt_migrated_dep[40][0] == 63

    # Inspect English / None tokens_40 row (row 3 untouched)
    assert df_migrated.at[3, "dep"] == [(1, "root")] * 10

    # Pass 2: Dry-run / idempotency check on already migrated data
    df_pass2, telem2 = adjust_dataframe_dep_chunks(df_migrated, chunk_size=40)
    assert telem2["rows_modified"] == 0
    assert telem2["already_global_chunks"] == 4  # YG (1), KT (1), 85-tok (2)
    assert telem2["heads_shifted"] == 0

    # Mixed-state error guard
    mixed_df = pd.DataFrame({
        "tokens_40": [[make_synthetic_tokens(40), make_synthetic_tokens(4)]],
        "dep": [[(1, "root")] * 40 + [(2, "dep"), (42, "dep"), (0, "root"), (0, "root")]],
    })
    with pytest.raises(ValueError, match="mixed/invalid heads"):
        adjust_dataframe_dep_chunks(mixed_df, chunk_size=40)
