from __future__ import annotations
from typing import Tuple
import torch  # type: ignore

_BASE_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def seq_to_onehot(seq: str, out_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Left-aligned one-hot (4,out_len) and mask (out_len,). Unknowns -> zero vec, mask=1."""
    onehot = torch.zeros((4, out_len), dtype=torch.float32)
    mask = torch.zeros((out_len,), dtype=torch.float32)
    upto = min(len(seq), out_len)
    for i in range(upto):
        idx = _BASE_TO_IDX.get(seq[i], None)
        if idx is not None:
            onehot[idx, i] = 1.0
        mask[i] = 1.0
    return onehot, mask


def seq_to_onehot_right_aligned(seq: str, out_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-aligned one-hot and mask."""
    onehot = torch.zeros((4, out_len), dtype=torch.float32)
    mask = torch.zeros((out_len,), dtype=torch.float32)
    L = len(seq)
    upto = min(L, out_len)
    pad = max(0, out_len - L)
    start = max(0, L - out_len)
    for i, ch in enumerate(seq[start:start + upto]):
        idx = _BASE_TO_IDX.get(ch, None)
        if idx is not None:
            onehot[idx, pad + i] = 1.0
        mask[pad + i] = 1.0
    return onehot, mask
