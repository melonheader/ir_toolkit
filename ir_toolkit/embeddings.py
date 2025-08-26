from __future__ import annotations
from typing import Dict
import os
import h5py
from tqdm.auto import tqdm # type: ignore

import torch  # type: ignore
import torch.nn.functional as F  # type: ignore

from .encoding import seq_to_onehot # type: ignore


def compute_and_save_embeddings(
    backbone,
    seq_dataset: Dict[str, str],
    output_path: str,
    pool_dim: int = 64,
    device: str = "cuda",
    dataset_name: str = "embeddings",
):
    """
    backbone(x) -> (B, feat_dim, L)
    Saves:
      - ids: (N,)
      - {dataset_name}: (N, feat_dim, pool_dim)
    """
    if os.path.exists(output_path):
        os.remove(output_path)

    backbone.eval().to(device)
    ids = list(seq_dataset.keys())

    with torch.no_grad(), h5py.File(output_path, "w") as h5f:
        id_ds = h5f.create_dataset("ids", (len(ids),), dtype=h5py.string_dtype())
        emb_ds = None

        for i, sid in enumerate(tqdm(ids, desc="Embedding", unit="seq")):
            seq = seq_dataset[sid]
            onehot, _ = seq_to_onehot(seq, len(seq))
            x = onehot.unsqueeze(0).to(device)  # (1,4,L)
            emb = backbone(x).squeeze(0)        # (C,L)
            pooled = F.adaptive_avg_pool1d(emb, pool_dim)  # (C,pool)

            if emb_ds is None:
                feat_dim = pooled.shape[0]
                emb_ds = h5f.create_dataset(dataset_name, shape=(len(ids), feat_dim, pool_dim), dtype="float32")

            emb_ds[i] = pooled.cpu().numpy()
            id_ds[i] = sid

    print(f"Embeddings saved to {output_path} in dataset '{dataset_name}'")
