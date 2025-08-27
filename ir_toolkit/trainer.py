from __future__ import annotations
from typing import Optional
import numpy as np

import torch  # type: ignore
import pytorch_lightning as pl  # type: ignore
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor  # type: ignore
from pytorch_lightning.loggers import TensorBoardLogger  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix  # type: ignore

from .models import IntronEndCAMModel, IntronEndLightning, LogCollector


def create_backbone_model():
    """Simple CNN backbone if none is provided."""
    import torch.nn as nn  # local alias # type: ignore
    class SimpleCNNBackbone(nn.Module):
        def __init__(self, input_dim=4, output_dim=768):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(input_dim, 128, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(128),
                nn.Conv1d(128, 256, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(256),
            )
            self.body = nn.Sequential(
                nn.Conv1d(256, 512, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(512),
                nn.Conv1d(512, output_dim, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(output_dim),
            )
        def forward(self, x):  # (B,4,L) -> (B, output_dim, L)
            return self.body(self.stem(x))
    return SimpleCNNBackbone()


def setup_training(
    train_loader,
    backbone_model=None,
    feat_dim=768,
    attn_hidden_dim=256,
    use_attention_pooling=False,
    use_joint_classifier=False,
    jc_head_hidden=16,
    use_simple_fusion=False,
    use_middle=False,
    middle_dim=0,
    lr=1e-4,
    weight_decay=1e-5,
    max_epochs=50,
    patience=10,
    val_check_interval=1.0,
    pos_weight=None,
    freeze_backbone_initial=True,
    unfreeze_after_epochs=5,
    experiment_name="splice_site_cam",
):
    if backbone_model is None:
        print("No backbone provided, using simple CNN backbone")
        backbone_model = create_backbone_model()

    if pos_weight is None:
        pos_count = 0
        neg_count = 0
        for batch in train_loader:
            *_, labels = batch
            pos_count += labels.sum().item()
            neg_count += (1 - labels).sum().item()
        pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        print(f"Calculated pos_weight: {pos_weight:.3f} (pos: {pos_count}, neg: {neg_count})")

    cam_model = IntronEndCAMModel(
        backbone=backbone_model,
        feat_dim=feat_dim,
        use_attention_pooling=use_attention_pooling,
        attn_hidden=attn_hidden_dim,
        use_simple_fusion=use_simple_fusion,
        use_joint_classifier=use_joint_classifier,
        jc_head_hidden=jc_head_hidden,
        use_middle=use_middle,
        middle_dim=middle_dim,
    )

    lightning_model = IntronEndLightning(
        model=cam_model,
        lr=lr,
        weight_decay=weight_decay,
        pos_weight=pos_weight,
        freeze_backbone_initial=freeze_backbone_initial,
        unfreeze_after_epochs=unfreeze_after_epochs,
    )

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{experiment_name}_{timestamp}"

    logger = TensorBoardLogger(save_dir="logs", name=f"{experiment_name}_{timestamp}", version=None)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{log_dir}/checkpoints",
        filename="best_model_{epoch:02d}_{val_auroc:.3f}",
        save_top_k=3,
        monitor="val_auroc",
        mode="max",
        save_last=True,
        verbose=False,
    )
    early_stop_callback = EarlyStopping(monitor="val_auroc", min_delta=0.001, patience=patience, mode="max")
    lr_monitor = LearningRateMonitor(logging_interval='epoch', log_weight_decay=True)
    log_collector = LogCollector()

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, log_collector],
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else "32",
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        val_check_interval=val_check_interval,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    return trainer, lightning_model, log_collector, checkpoint_callback


def run_training(trainer, model, train_loader, val_loader):
    print("Starting training...")
    print(f"Training batches: {len(train_loader)} | Validation batches: {len(val_loader)}")
    trainer.fit(model, train_loader, val_loader)
    print("Training completed!")
    return trainer.callback_metrics


def evaluate_model(model, val_loader, device: Optional[str] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    all_preds, all_labels, all_ids = [], [], []
    all_cam_left, all_cam_right = [], []

    with torch.no_grad():
        for batch in val_loader:
            ids, left, left_mask, right, right_mask, middle, labels = batch
            left, left_mask = left.to(device), left_mask.to(device)
            right, right_mask = right.to(device), right_mask.to(device)
            labels = labels.to(device)
            if middle is not None:
                middle = middle.to(device)

            outputs = model(left, left_mask, right, right_mask, middle)
            probs = torch.sigmoid(outputs['seq_logit'])
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_ids.extend(ids)
            all_cam_left.append(outputs['cam_left'].cpu())
            all_cam_right.append(outputs['cam_right'].cpu())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_cam_left = torch.cat(all_cam_left, dim=0)
    all_cam_right = torch.cat(all_cam_right, dim=0)
    binary_preds = (all_preds > 0.5).astype(int)

    print("Classification Report:")
    print(classification_report(all_labels, binary_preds))

    cm = confusion_matrix(all_labels, binary_preds)
    print("\nConfusion Matrix:")
    print(cm)

    return {
        'predictions': all_preds,
        'labels': all_labels,
        'ids': all_ids,
        'cam_left': all_cam_left,
        'cam_right': all_cam_right,
        'binary_predictions': binary_preds,
        'confusion_matrix': cm,
    }
