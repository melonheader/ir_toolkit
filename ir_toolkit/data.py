from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple


def prepare_dataset(
    source,
    class_col: str,
    class_lims: Tuple[float, float],
    drop_mid: bool = True,
    extra_cols: Optional[List[str]] = None,
    col_filters: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """
    Build a {EVENT -> class} DataFrame from a CSV/DataFrame.
    class_lims = (low, high); class=1 if >= high, class=0 if < low, else 'else'.
    """
    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f'Path does not exist: {source}')
        source = pd.read_csv(source)
    elif not isinstance(source, pd.DataFrame):
        raise ValueError('source must be a path or a pandas DataFrame')

    if class_col not in source.columns:
        raise ValueError(f'class column "{class_col}" not found')
    if not (isinstance(class_lims, tuple) and len(class_lims) == 2):
        raise ValueError('class_lims must be a (low, high) tuple')

    source = source.dropna(subset=[class_col]).copy().reset_index(drop=True)

    if col_filters:
        for col, (low, high) in col_filters.items():
            if col not in source.columns:
                print(f'Column "{col}" not found; skipping filter.')
                continue
            before = len(source)
            source = source[(source[col] >= low) & (source[col] < high)].copy()
            print(f'Filtered "{col}" in [{low}, {high}): {before} -> {len(source)} rows')

    if 'EVENT' not in source.columns:
        raise ValueError('Expected an "EVENT" column to use as index.')

    low, high = class_lims
    class_conds = [source[class_col] >= high, source[class_col] < low]
    class_names = ['1', '0']
    source['class'] = np.select(class_conds, class_names, default='else').astype(str)
    if drop_mid:
        source = source[source['class'] != 'else'].copy()

    out_cols = ['class']
    if extra_cols:
        out_cols += [c for c in extra_cols if c in source.columns]
    source = source.set_index(source['EVENT'].astype(str))[out_cols].copy()
    return source


def load_fasta_to_dict(
    fasta_path: str,
    ids_keep: Optional[set] = None,
    id_func=lambda x: x,
) -> Dict[str, str]:
    """Reads FASTA into {id: sequence}. Applies id_func to header token."""
    seqs: Dict[str, str] = {}
    with open(fasta_path, 'r') as fh:
        cur_id, cur_seq = None, []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if cur_id is not None:
                    seq = ''.join(cur_seq).upper()
                    if ids_keep is None or cur_id in ids_keep:
                        seqs[cur_id] = seq
                cur_id = id_func(line[1:].split()[0])
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id is not None:
            seq = ''.join(cur_seq).upper()
            if ids_keep is None or cur_id in ids_keep:
                seqs[cur_id] = seq
    return seqs
