from __future__ import annotations
from typing import Optional, Dict, List
import numpy as np
import h5py
from tqdm.auto import tqdm  # type: ignore

import torch  # type: ignore
from torch.utils.data import Dataset  # type: ignore

from .encoding import seq_to_onehot, seq_to_onehot_right_aligned


class IntronsEndsDataset(Dataset):
    """
    Returns a dict with left/right one-hot/mask, optional middle vector, and label.
    """
    def __init__(
        self,
        ids: List[str],
        fasta_seqs: Dict[str, str],
        df,
        L_left: int = 512,
        L_right: int = 512,
        use_middle: bool = False,
        middle_embeddings_h5: Optional[str] = None,
        middle_dataset_name: str = "embeddings",
    ):
        self.fasta_seqs = fasta_seqs
        missing = [sid for sid in ids if sid not in fasta_seqs]
        if missing:
            print(f"Warning: {len(missing)} ids not in FASTA; they will be skipped.")

        self.ids = [sid for sid in ids if sid in fasta_seqs]
        self.id_to_idx = {sid: i for i, sid in enumerate(self.ids)}
        self.df = df
        self.L_left = L_left
        self.L_right = L_right

        self.use_middle = use_middle
        self.h5f: Optional[h5py.File] = None
        self.emb_idx_map: Optional[Dict[str, int]] = None
        self.middle_dataset_name = middle_dataset_name

        if self.use_middle and middle_embeddings_h5 is not None:
            self.h5f = h5py.File(middle_embeddings_h5, "r")
            emb_ids = [x.decode() if isinstance(x, bytes) else x for x in self.h5f["ids"]]
            self.emb_idx_map = {sid: i for i, sid in enumerate(emb_ids)}
        elif self.use_middle and middle_embeddings_h5 is None:
            print("use_middle=True but no H5 provided; proceeding with use_middle=False")
            self.use_middle = False

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            idx = self.id_to_idx[idx]
        seq_id = self.ids[idx]
        seq = self.fasta_seqs[seq_id]

        label = float(self.df.loc[seq_id, 'class'])
        label = torch.tensor(label, dtype=torch.float32)

        left_ohe, left_mask = seq_to_onehot(seq, self.L_left)
        right_ohe, right_mask = seq_to_onehot_right_aligned(seq, self.L_right)

        middle_bins = None
        if self.use_middle:
            if seq_id not in self.emb_idx_map:
                raise KeyError(f"{seq_id} not found in embeddings HDF5")
            emb_idx = self.emb_idx_map[seq_id]
            vec = np.asarray(self.h5f[self.middle_dataset_name][emb_idx], dtype=np.float32).reshape(-1)
            middle_bins = torch.from_numpy(vec)

        return {
            "id": seq_id,
            "left_onehot": left_ohe,
            "left_mask": left_mask,
            "right_onehot": right_ohe,
            "right_mask": right_mask,
            "middle_bins": middle_bins,  # 1D or None
            "label": label,
        }


def introns_collate_fn(batch):
    ids = [b["id"] for b in batch]
    left = torch.stack([b["left_onehot"] for b in batch], 0)
    left_mask = torch.stack([b["left_mask"] for b in batch], 0)
    right = torch.stack([b["right_onehot"] for b in batch], 0)
    right_mask = torch.stack([b["right_mask"] for b in batch], 0)
    labels = torch.stack([b["label"] for b in batch], 0)
    middles = None
    if batch[0]["middle_bins"] is not None:
        middles = torch.stack([b["middle_bins"] for b in batch], 0)
    return ids, left, left_mask, right, right_mask, middles, labels
