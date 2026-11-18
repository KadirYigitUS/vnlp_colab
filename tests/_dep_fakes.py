"""Deterministic fake dependency parser and test fixtures for unit tests.

Provides synthetic token sequences and deterministic dependency parser fakes
allowing pure-Python testing of chunking, merging, and pipeline invariants
without loading Keras 3 neural network weights.
"""

from typing import List, Sequence, Tuple

from vnlp_colab.dep.dep_utils_colab import (
    SENTENCE_MAX_LEN,
    chunk_token_sequence,
    merge_dep_chunks,
)


def make_synthetic_tokens(count: int, prefix: str = "tok") -> List[str]:
    """Generate a deterministic token list of specified length."""
    return [f"{prefix}_{i + 1}" for i in range(count)]


class DeterministicFakeDP:
    """Deterministic fake Dependency Parser mirroring SPUContextDP/TreeStackDP chunking."""

    def __init__(self, model_name: str = "FakeSPUContextDP", max_len: int = SENTENCE_MAX_LEN):
        self.model_name = model_name
        self.max_len = max_len

    def _parse_single_chunk(self, chunk: Sequence[str]) -> List[Tuple[int, str, int, str]]:
        """Return 4-tuples with 1-based local id, root at token 1, others pointing to 1."""
        res: List[Tuple[int, str, int, str]] = []
        for idx, tok in enumerate(chunk):
            local_id = idx + 1
            if idx == 0:
                head, rel = 0, "root"
            else:
                head, rel = 1, "dep"
            res.append((local_id, tok, head, rel))
        return res

    def predict(self, tokens: List[str]) -> List[Tuple[int, str, int, str]]:
        """Predict dependencies for a single token list."""
        if not tokens:
            return []
        return self.predict_batch([tokens])[0]

    def predict_batch(
        self, batch_of_tokens: List[List[str]]
    ) -> List[List[Tuple[int, str, int, str]]]:
        """Predict dependencies for a batch of token lists."""
        results: List[List[Tuple[int, str, int, str]]] = []
        for tokens in batch_of_tokens:
            if not tokens:
                results.append([])
                continue
            chunks = chunk_token_sequence(tokens, self.max_len)
            parsed_chunks = [self._parse_single_chunk(c) for c in chunks]
            merged = merge_dep_chunks(parsed_chunks, chunks)
            results.append(merged)
        return results
