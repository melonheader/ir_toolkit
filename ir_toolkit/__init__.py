"""
ir_toolkit: Tools for intron retention training and analysis.

Public API is exposed lazily to keep `import ir_trainer` lightweight.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

__version__ = "0.1.0"

# Lazy attribute loader to avoid importing heavy submodules at package import time
def __getattr__(name: str):
    # light
    if name in {"prepare_dataset", "load_fasta_to_dict"}:
        from . import data as _m
        return getattr(_m, name)

    # encoding / datasets / embeddings (torch)
    if name in {"seq_to_onehot", "seq_to_onehot_right_aligned"}:
        from . import encoding as _m
        return getattr(_m, name)

    if name in {"IntronsEndsDataset", "introns_collate_fn"}:
        from . import datasets as _m
        return getattr(_m, name)

    if name in {"compute_and_save_embeddings"}:
        from . import embeddings as _m
        return getattr(_m, name)

    # models (torch + lightning)
    if name in {
        "ParnetBackbone", "SpliceAIBackbone", "load_spliceAI_model",
        "CAMPerPositionHead", "IntronEndCAMModel", "IntronEndLightning", "LogCollector",
    }:
        from . import models as _m
        return getattr(_m, name)

    # trainer (lightning + sklearn)
    if name in {"create_backbone_model", "setup_training", "run_training", "evaluate_model"}:
        from . import trainer as _m
        return getattr(_m, name)

    # viz (matplotlib + sklearn)
    if name in {"plot_training_history", "visualize_cam_examples", "plot_roc_prc"}:
        from . import viz as _m
        return getattr(_m, name)

    raise AttributeError(f"module 'ir_trainer' has no attribute {name!r}")

# For static analysis & IDEs (no runtime import)
if TYPE_CHECKING:
    from .data import prepare_dataset, load_fasta_to_dict
    from .encoding import seq_to_onehot, seq_to_onehot_right_aligned
    from .datasets import IntronsEndsDataset, introns_collate_fn
    from .embeddings import compute_and_save_embeddings
    from .models import (
        ParnetBackbone, SpliceAIBackbone, load_spliceAI_model,
        CAMPerPositionHead, IntronEndCAMModel, IntronEndLightning, LogCollector,
    )
    from .trainer import create_backbone_model, setup_training, run_training, evaluate_model
    from .viz import plot_training_history, visualize_cam_examples, plot_roc_prc

__all__ = [
    # light
    "prepare_dataset", "load_fasta_to_dict",
    # encoding / datasets / embeddings
    "seq_to_onehot", "seq_to_onehot_right_aligned",
    "IntronsEndsDataset", "introns_collate_fn",
    "compute_and_save_embeddings",
    # models
    "ParnetBackbone", "SpliceAIBackbone", "load_spliceAI_model",
    "CAMPerPositionHead", "IntronEndCAMModel", "IntronEndLightning", "LogCollector",
    # trainer
    "create_backbone_model", "setup_training", "run_training", "evaluate_model",
    # viz
    "plot_training_history", "visualize_cam_examples", "plot_roc_prc",
    "__version__",
]
