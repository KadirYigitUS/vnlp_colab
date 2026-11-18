"""Idempotent dependency head offset migration for multi-chunk VNLP DataFrames.

Shifts chunk-local heads in chunk k >= 1 by + chunk_size * k, preserving
roots (head == 0), clamping out-of-bounds heads, and maintaining idempotency.
"""

import ast
import logging
from typing import Any, Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def _parse_list_field(val: Any) -> Any:
    """Safely parse str representation of a list or return as-is."""
    if isinstance(val, str):
        val = val.strip()
        if (val.startswith("[") and val.endswith("]")) or (val.startswith("(") and val.endswith(")")):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"Failed to parse list literal: {e}") from e
    return val


def adjust_dataframe_dep_chunks(
    df: pd.DataFrame, chunk_size: int = 40
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Adjust dependency head offsets for multi-chunk sentences in a DataFrame.

    Idempotently shifts local 1..chunk_size heads in chunk k >= 1 by + chunk_size * k.
    Clamps any local_head > chunk_len to chunk_len and records dep_head_oob.
    Skips English spaCy DataFrames (no tokens_40) and single-chunk rows (<= 40 tokens).
    """
    telemetry: Dict[str, int] = {
        "rows_inspected": len(df),
        "multi_chunk_rows": 0,
        "rows_modified": 0,
        "heads_shifted": 0,
        "dep_head_oob": 0,
        "already_global_chunks": 0,
        "all_root_chunks": 0,
    }

    if "tokens_40" not in df.columns or "dep" not in df.columns:
        return df, telemetry

    df_out = df.copy()

    for idx in df_out.index:
        t40_raw = df_out.at[idx, "tokens_40"]
        if t40_raw is None or (not isinstance(t40_raw, (list, tuple, dict, str)) and pd.isna(t40_raw)):
            continue

        t40 = _parse_list_field(t40_raw)
        if not isinstance(t40, list) or len(t40) <= 1:
            continue

        telemetry["multi_chunk_rows"] += 1
        dep_raw = df_out.at[idx, "dep"]
        is_str_dep = isinstance(dep_raw, str)
        dep = _parse_list_field(dep_raw)

        if not isinstance(dep, list):
            raise ValueError(f"Row {idx}: expected list for dep, got {type(dep)}")

        chunk_lens = [len(c) for c in t40]
        total_tokens = sum(chunk_lens)
        if len(dep) != total_tokens:
            raise ValueError(
                f"Row {idx}: dep length {len(dep)} does not match tokens_40 total {total_tokens}"
            )

        dep_chunks: List[List[Any]] = []
        offset = 0
        for clen in chunk_lens:
            dep_chunks.append(list(dep[offset : offset + clen]))
            offset += clen

        if len(dep_chunks) != len(t40):
            raise ValueError(f"Row {idx}: dep_chunks count mismatch {len(dep_chunks)} != {len(t40)}")

        row_modified = False
        new_dep: List[Any] = list(dep_chunks[0])

        for k in range(1, len(t40)):
            chunk_dep = dep_chunks[k]
            chunk_len = chunk_lens[k]
            nonzero_heads = [h for (h, _) in chunk_dep if h != 0]

            if not nonzero_heads:
                telemetry["all_root_chunks"] += 1
                new_dep.extend(chunk_dep)
                continue

            is_chunk_local = all(1 <= h <= chunk_size for h in nonzero_heads)
            is_already_global = all(h > chunk_size * k for h in nonzero_heads)

            if is_already_global:
                telemetry["already_global_chunks"] += 1
                new_dep.extend(chunk_dep)
            elif is_chunk_local:
                row_modified = True
                shifted_chunk: List[Tuple[int, str]] = []
                for local_head, rel in chunk_dep:
                    if local_head == 0:
                        shifted_chunk.append((0, rel))
                    else:
                        if local_head > chunk_len:
                            logger.warning(
                                "dep_head_oob at row %s chunk %d: head=%d > chunk_len=%d",
                                idx,
                                k,
                                local_head,
                                chunk_len,
                            )
                            telemetry["dep_head_oob"] += 1
                            local_head = chunk_len

                        global_head = local_head + chunk_size * k
                        telemetry["heads_shifted"] += 1
                        shifted_chunk.append((global_head, rel))
                new_dep.extend(shifted_chunk)
            else:
                raise ValueError(
                    f"Row {idx} chunk {k} has mixed/invalid heads: {nonzero_heads} "
                    f"(expected all in [1, {chunk_size}] or all > {chunk_size * k})"
                )

        if row_modified:
            telemetry["rows_modified"] += 1
            df_out.at[idx, "dep"] = repr(new_dep) if is_str_dep else new_dep

    return df_out, telemetry
