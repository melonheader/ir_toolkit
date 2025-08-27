"""
ir_toolkit: intron-retention training & analysis.

This __init__ eagerly re-exports public API so that IDEs/Jupyter show
signatures and docstrings directly on `import ir_toolkit as irt`.
"""

from __future__ import annotations

# ---- Version ---------------------------------------------------------------
__version__ = "0.1.0"

# ---- Light (no heavy deps) ------------------------------------------------
from .data import (
    prepare_dataset,
    load_fasta_to_dict,
)

# ---- Torch-based (Dataset/encoding/embeddings) ----------------------------
from .encoding import (
    seq_to_onehot,
    seq_to_onehot_right_aligned,
)

from .datasets import (
    IntronsEndsDataset,
    introns_collate_fn,
)

from .embeddings import (
    compute_and_save_embeddings,
)

# ---- Models / Lightning ----------------------------------------------------
from .models import (
    ParnetBackbone,
    SpliceAIBackbone,
    load_spliceAI_model,
    CAMPerPositionHead,
    IntronEndCAMModel,
    IntronEndLightning,
    LogCollector,
)

# ---- Trainer orchestration -------------------------------------------------
from .trainer import (
    create_backbone_model,
    setup_training,
    run_training,
    evaluate_model,
)

# ---- Visualization ---------------------------------------------------------
from .viz import (
    plot_training_history,
    visualize_cam_examples,
    plot_roc_prc,
)

# ---- Public API surface ----------------------------------------------------
__all__ = [
    "__version__",
    # data
    "prepare_dataset", "load_fasta_to_dict",
    # encoding
    "seq_to_onehot", "seq_to_onehot_right_aligned",
    # datasets
    "IntronsEndsDataset", "introns_collate_fn",
    # embeddings
    "compute_and_save_embeddings",
    # models
    "ParnetBackbone", "SpliceAIBackbone", "load_spliceAI_model",
    "CAMPerPositionHead", "IntronEndCAMModel", "IntronEndLightning", "LogCollector",
    # trainer
    "create_backbone_model", "setup_training", "run_training", "evaluate_model",
    # viz
    "plot_training_history", "visualize_cam_examples", "plot_roc_prc",
]
