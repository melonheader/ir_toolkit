from __future__ import annotations
from typing import Optional, Dict
import numpy as np

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F # type: ignore
import pytorch_lightning as pl  # type: ignore
from torchmetrics.classification import AUROC, Accuracy  # type: ignore

# def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-8):
#     # mask: (B,L) in {0,1}
#     mask = mask.float()
#     logits = logits.masked_fill(mask == 0, -1e9)
#     logits = logits - logits.max(dim=dim, keepdim=True).values
#     exps = torch.exp(logits) * mask
#     den = exps.sum(dim=dim, keepdim=True).clamp(min=eps)
#     return exps / den

def load_spliceAI_model(weights_path: str, flanking_size: int = 400, SL: int = 256, device: str = 'cuda'):
    """
    Load OpenSpliceAI model and weights.

    Requires manual installation of OpenSpliceAI:
        git clone https://github.com/Kuanhao-Chao/OpenSpliceAI.git
        cd OpenSpliceAI
        python setup.py install
    """
    try:
        from openspliceai.train_base.openspliceai import SpliceAI  # type: ignore
    except ImportError as e:
        raise ImportError(
            "OpenSpliceAI is not installed.\n\n"
            "Install it manually with:\n"
            "    git clone https://github.com/Kuanhao-Chao/OpenSpliceAI.git\n"
            "    cd OpenSpliceAI\n"
            "    python setup.py install\n\n"
            "After installation, re-run your code."
        ) from e
    L = 32
    if flanking_size == 80:
        W = np.asarray([11, 11, 11, 11]); AR = np.asarray([1, 1, 1, 1])
    elif flanking_size == 400:
        W = np.asarray([11]*8); AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
    elif flanking_size == 2000:
        W = np.asarray([11]*8 + [21]*4); AR = np.asarray([1]*8 + [10]*4)
    elif flanking_size == 10000:
        W = np.asarray([11]*8 + [21]*4 + [41]*4); AR = np.asarray([1]*8 + [10]*4 + [25]*4)
    else:
        raise ValueError(f"Unsupported flanking_size: {flanking_size}")

    CL = int(2 * np.sum(AR * (W - 1)))
    print(f"[INFO] Context: {CL} nt | Output SL: {SL}")

    model = SpliceAI(L, W, AR).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    return model


class ParnetBackbone(nn.Module):
    """Wrap an RBPNet-like model exposing .stem and .body."""
    def __init__(self, rbpnet: nn.Module):
        super().__init__()
        self.stem = rbpnet.stem
        self.body = rbpnet.body

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(self.stem(x))  # (B,C,L)


class SpliceAIBackbone(nn.Module):
    """Wrap an OpenSpliceAI model, optionally cropping."""
    def __init__(self, spliceai: nn.Module, cropping: bool = False):
        super().__init__()
        self.initial_conv = spliceai.initial_conv
        self.initial_skip = spliceai.initial_skip
        self.residual_units = spliceai.residual_units
        self.crop = spliceai.crop if cropping else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        x, skip = self.initial_skip(x, 0)
        for m in self.residual_units:
            x, skip = m(x, skip)
        if self.crop is not None:
            skip = self.crop(skip)
        return skip


class CAMPerPositionHead(nn.Module):
    """Per-position CAM + sequence logit."""
    def __init__(self, feat_dim=768, use_attention_pooling=False, attn_hidden=128):
        super().__init__()
        self.use_attention_pooling = use_attention_pooling
        if use_attention_pooling:
            self.attention_pooling = nn.Sequential(
                nn.Conv1d(feat_dim, attn_hidden, 1),
                nn.Tanh(),
                nn.Conv1d(attn_hidden, 1, 1),
            )
        self.gap_classifier = nn.Linear(feat_dim, 1, bias=True)

    def forward(self, feats: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, C, L = feats.shape
        if mask is None:
            mask = torch.ones(B, L, device=feats.device)

        if self.use_attention_pooling:
            # attn_logits = self.attention_pooling(feats).squeeze(1)
            # attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))
            # attn_weights = torch.softmax(attn_logits, dim=-1)
            # #attn_weights = _masked_softmax(attn_logits, mask, dim=-1) 
            # pooled = torch.einsum('bcl,bl->bc', feats, attn_weights)
            ##
            attn_logits = self.attention_pooling(feats).squeeze(1)      # (B, L)
            neg_large = torch.finfo(attn_logits.dtype).min
            x = attn_logits.masked_fill(mask == 0, neg_large)
            attn_weights = torch.softmax(x, dim=-1)
            # strict zero + renorm:
            attn_weights = attn_weights * mask.to(attn_logits.dtype)
            attn_weights = attn_weights / attn_weights.sum(-1, keepdim=True).clamp_min(1e-8)

            pooled = torch.einsum('bcl,bl->bc', feats, attn_weights)

        else:
            attn_weights = None
            mexp = mask.unsqueeze(1)
            pooled = (feats * mexp).sum(-1) / mexp.sum(-1).clamp(min=1e-8)

        seq_logit = self.gap_classifier(pooled).squeeze(-1)
        w = self.gap_classifier.weight.squeeze(0)
        cam = torch.einsum('c,bcl->bl', w, feats).masked_fill(mask == 0, 0.0)
        return cam, seq_logit, attn_weights

    def get_feature_importance(self):
        return self.gap_classifier.weight.detach().cpu().numpy().squeeze()


class IntronEndCAMModel(nn.Module):
    """
    Shared backbone (left/right) + CAM heads; optional middle vector; fusion by
    joint MLP, simple linear, or mean.
    """
    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int = 768,
        use_attention_pooling: bool = False,
        attn_hidden: int = 128,
        use_middle: bool = False,
        middle_dim: int = 0,
        use_joint_classifier: bool = False,
        use_simple_fusion: bool = False,
        jc_head_hidden: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = backbone
        self.use_middle = use_middle
        self.use_joint_classifier = use_joint_classifier
        self.use_simple_fusion = use_simple_fusion

        self.left_cam_head = CAMPerPositionHead(feat_dim, use_attention_pooling, attn_hidden)
        self.right_cam_head = CAMPerPositionHead(feat_dim, use_attention_pooling, attn_hidden)

        if self.use_middle:
            if middle_dim <= 0:
                raise ValueError("use_middle=True requires middle_dim > 0")
            self.middle_head = nn.Linear(middle_dim, 1)

        fdim = 2 + (1 if self.use_middle else 0)
        if self.use_joint_classifier:
            self.joint_classifier = nn.Sequential(
                nn.Linear(fdim, jc_head_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(jc_head_hidden, 1)
            )
        elif self.use_simple_fusion:
            self.simple_fusion = nn.Linear(fdim, 1)

    def forward(
        self,
        left_onehot: torch.Tensor,
        left_mask: torch.Tensor,
        right_onehot: torch.Tensor,
        right_mask: torch.Tensor,
        middle_bins: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        left_feats = self.backbone(left_onehot)
        cam_left, logit_left, attn_left = self.left_cam_head(left_feats, left_mask)

        right_feats = self.backbone(right_onehot)
        cam_right, logit_right, attn_right = self.right_cam_head(right_feats, right_mask)

        logits = [logit_left, logit_right]
        logit_middle = None
        if self.use_middle:
            if middle_bins is None:
                raise ValueError("use_middle=True but middle_bins=None")
            if middle_bins.dim() > 2:
                middle_bins = middle_bins.reshape(middle_bins.size(0), -1)
            logit_middle = self.middle_head(middle_bins).squeeze(-1)
            logits.append(logit_middle)

        joint = torch.stack(logits, dim=1)
        if self.use_joint_classifier:
            seq_logit = self.joint_classifier(joint).squeeze(-1)
        elif self.use_simple_fusion:
            seq_logit = self.simple_fusion(joint).squeeze(-1)
        else:
            seq_logit = joint.mean(dim=1)

        return {
            "seq_logit": seq_logit,
            "logit_left": logit_left,
            "logit_right": logit_right,
            "logit_middle": logit_middle if self.use_middle else None,
            "cam_left": cam_left,
            "cam_right": cam_right,
            "attn_left": attn_left,
            "attn_right": attn_right,
        }

    def get_feature_importance_weights(self):
        return {
            "left_weights": self.left_cam_head.get_feature_importance(),
            "right_weights": self.right_cam_head.get_feature_importance(),
        }


class IntronEndLightning(pl.LightningModule):
    def __init__(
        self,
        model: IntronEndCAMModel,
        lr = 1e-4,              # head LR (joint/simple heads)
        backbone_lr_base = 1e-5,        # base LR for backbone (max for deepest layer)
        llrd_gamma = 0.7,
        weight_decay: float = 1e-6,
        pos_weight: Optional[float] = None,
        freeze_backbone_initial: bool = True,
        unfreeze_after_epochs: Optional[int] = 5,
        scheduler_patience: int = 3,
        scheduler_factor: float = 0.5,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.backbone_lr_base = backbone_lr_base
        self.llrd_gamma = llrd_gamma
        self.weight_decay = weight_decay
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.save_hyperparameters(ignore=['model'])

        if pos_weight is not None:
            self.register_buffer('pos_weight', torch.tensor(pos_weight, dtype=torch.float32))
        else:
            self.pos_weight = None

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self._val_preds = []
        self._val_tgts  = []
        ##
        self.freeze_backbone_initial = freeze_backbone_initial
        self.unfreeze_after_epochs = unfreeze_after_epochs
        if freeze_backbone_initial:
            for p in self.model.backbone.parameters():
                p.requires_grad = False
    def _is_no_decay(self, name: str, p: torch.nn.Parameter) -> bool:
        return (p.ndim == 1) or name.endswith(".bias") or ("bn" in name.lower()) or ("norm" in name.lower())

    def _backbone_layers_in_order(self):
        # returns list of (layer_name, module) in shallow→deep order
        bb = self.model.backbone
        layers = [("model.backbone.stem", bb.stem)]
        for i, m in enumerate(bb.body):
            layers.append((f"model.backbone.body.{i}", m))
        return layers
    
    def forward(self, left, left_mask, right, right_mask, middle_vec=None):
        return self.model(left, left_mask, right, right_mask, middle_vec)

    # def on_train_epoch_start(self):
    #     if self.freeze_backbone_initial and self.unfreeze_after_epochs is not None:
    #         if self.current_epoch == self.unfreeze_after_epochs:
    #             for p in self.model.backbone.parameters():
    #                 p.requires_grad = True
    #             # turn on lr for backbone params
    #             optim = self.trainer.optimizers[0]
    #             bb_params = set(self.model.backbone.parameters())
    #             for g in optim.param_groups:
    #                 if any(p in bb_params for p in g["params"]):
    #                     g["lr"] = self.lr
    #             print(f"[INFO] Unfroze backbone and enabled LR at epoch {self.current_epoch}")

    def on_train_epoch_start(self):
        # staged unfreeze: switch BB param-group LRs from 0 → LLRD schedule
        if self.freeze_backbone_initial and self.unfreeze_after_epochs is not None:
            if self.current_epoch == self.unfreeze_after_epochs:
                for p in self.model.backbone.parameters():
                    p.requires_grad = True

                # update optimizer param-group LRs according to LLRD now
                optim = self.trainer.optimizers[0]
                layers = self._backbone_layers_in_order()
                L = len(layers)

                # walk groups that contain backbone params and set LR per depth
                for k, (lname, lmod) in enumerate(layers):
                    lr_k = self.backbone_lr_base * (self.llrd_gamma ** (L - 1 - k))
                    layer_params = {p for _, p in lmod.named_parameters(recurse=True)}
                    for g in optim.param_groups:
                        if any(p in layer_params for p in g["params"]):
                            g["lr"] = float(lr_k)

                self.freeze_backbone_initial = False
                print(f"[INFO] Unfroze backbone at epoch {self.current_epoch}; "
                    f"BB LRs in [{self.backbone_lr_base*(self.llrd_gamma**(L-1)):.2e}, {self.backbone_lr_base:.2e}]")


    def training_step(self, batch, batch_idx):
        ids, left, left_mask, right, right_mask, middle, labels = batch
        out = self(left, left_mask, right, right_mask, middle)
        logits = out['seq_logit']
        loss = self.criterion(logits, labels.float())
        probs = torch.sigmoid(logits)

        self.train_auroc(probs, labels.long())
        self.train_acc((probs > 0.5).long(), labels.long())

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_auroc', self.train_auroc, on_step=False, on_epoch=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        return loss
    
    def on_validation_epoch_start(self):
        self.val_auroc.reset()
        self.val_acc.reset()
        self._val_preds.clear()
        self._val_tgts.clear()

    def validation_step(self, batch, batch_idx):
        ids, left, left_mask, right, right_mask, middle, labels = batch
        out = self(left, left_mask, right, right_mask, middle)
        logits = out['seq_logit']
        loss = self.criterion(logits, labels.float())
        probs = torch.sigmoid(logits)

        # accumulate; do not update
        self._val_preds.append(probs.detach())
        self._val_tgts.append(labels.detach())

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        preds = torch.cat(self._val_preds)
        tgts  = torch.cat(self._val_tgts).long()
        self.val_auroc.update(preds, tgts)
        self.val_acc.update((preds > 0.5).long(), tgts)
        self.log('val_auroc', self.val_auroc.compute(), prog_bar=True)
        self.log('val_acc',   self.val_acc.compute(),   prog_bar=True)

    # def configure_optimizers(self):
    #     # split by decay/no_decay but INCLUDE backbone regardless of requires_grad
    #     decay, no_decay = [], []
    #     for name, p in self.named_parameters():
    #         if p.ndim == 1 or name.endswith(".bias") or "bn" in name.lower() or "norm" in name.lower():
    #             no_decay.append(p)
    #         else:
    #             decay.append(p)

    #     optim = torch.optim.AdamW(
    #         [
    #             {"params": decay,    "weight_decay": self.weight_decay, "lr": self.lr},
    #             {"params": no_decay, "weight_decay": 0.0,               "lr": self.lr},
    #         ],
    #         lr=self.lr,
    #     )
    #     # set its param group lr to 0
    #     if self.freeze_backbone_initial:
    #         bb_params = set(self.model.backbone.parameters())
    #         for g in optim.param_groups:
    #             if any(p in bb_params for p in g["params"]):
    #                 g["lr"] = 0.0

    #     sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optim, mode="min", factor=self.scheduler_factor, patience=self.scheduler_patience
    #     )
    #     return {"optimizer": optim, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}

    def configure_optimizers(self):
        # 1) split HEAD vs BACKBONE named params
        head_named, bb_named = [], []
        for n, p in self.named_parameters():
            if n.startswith("model.backbone."):
                bb_named.append((n, p))
            else:
                head_named.append((n, p))

        # 2) HEAD groups (single LR)
        head_decay   = [p for n, p in head_named if not self._is_no_decay(n, p)]
        head_nodecay = [p for n, p in head_named if     self._is_no_decay(n, p)]
        param_groups = []
        if head_decay:
            param_groups.append({"params": head_decay,   "lr": self.lr, "weight_decay": self.weight_decay})
        if head_nodecay:
            param_groups.append({"params": head_nodecay, "lr": self.lr, "weight_decay": 0.0})

        # 3) BACKBONE groups (LLRD)
        layers = self._backbone_layers_in_order()
        L = len(layers)
        for k, (lname, lmod) in enumerate(layers):
            # deeper layers (higher k) get larger LR ≈ backbone_lr_base * (gamma ** (L-1-k))
            lr_k = self.backbone_lr_base * (self.llrd_gamma ** (L - 1 - k))
            # when frozen initially, start at 0.0
            if self.freeze_backbone_initial:
                lr_k = 0.0

            named = [(f"{lname}.{n}", p) for n, p in lmod.named_parameters(recurse=True)]
            if not named:
                continue
            decay   = [p for n, p in named if not self._is_no_decay(n, p)]
            nodecay = [p for n, p in named if     self._is_no_decay(n, p)]
            if decay:
                param_groups.append({"params": decay,   "lr": lr_k, "weight_decay": self.weight_decay})
            if nodecay:
                param_groups.append({"params": nodecay, "lr": lr_k, "weight_decay": 0.0})

        optim = torch.optim.AdamW(param_groups)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", factor=self.scheduler_factor, patience=self.scheduler_patience
        )
        return {"optimizer": optim, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}



class LogCollector(pl.Callback):
    def __init__(self):
        self.history = []

    def on_train_epoch_end(self, trainer, pl_module):
        rec = {'epoch': trainer.current_epoch, 'step': trainer.global_step}
        for k, v in trainer.logged_metrics.items():
            rec[k] = v.item() if torch.is_tensor(v) else v
        self.history.append(rec)

    def get_history_df(self):
        import pandas as pd
        return pd.DataFrame(self.history)
