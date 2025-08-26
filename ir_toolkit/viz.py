from __future__ import annotations
import numpy as np  # noqa
import matplotlib.pyplot as plt  # type: ignore


def plot_training_history(log_collector, save_path=None, show=True):
    history_df = log_collector.get_history_df()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    if {'train_loss_epoch', 'val_loss'}.issubset(history_df.columns):
        axes[0, 0].plot(history_df['epoch'], history_df['train_loss_epoch'], label='Train Loss')
        axes[0, 0].plot(history_df['epoch'], history_df['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss'); axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend(); axes[0, 0].grid(True)

    if {'train_auroc', 'val_auroc'}.issubset(history_df.columns):
        axes[0, 1].plot(history_df['epoch'], history_df['train_auroc'], label='Train AUROC')
        axes[0, 1].plot(history_df['epoch'], history_df['val_auroc'], label='Val AUROC')
        axes[0, 1].set_title('Training and Validation AUROC'); axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('AUROC')
        axes[0, 1].legend(); axes[0, 1].grid(True)

    if {'train_acc', 'val_acc'}.issubset(history_df.columns):
        axes[1, 0].plot(history_df['epoch'], history_df['train_acc'], label='Train Acc')
        axes[1, 0].plot(history_df['epoch'], history_df['val_acc'], label='Val Acc')
        axes[1, 0].set_title('Training and Validation Accuracy'); axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend(); axes[1, 0].grid(True)

    lr_cols = [c for c in history_df.columns if c.startswith("lr")]
    if lr_cols:
        for col in lr_cols:
            axes[1, 1].plot(history_df['epoch'], history_df[col], label=col)
        axes[1, 1].set_title('Learning Rate'); axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log'); axes[1, 1].legend(); axes[1, 1].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_cam_examples(results, sequences_dict, num_examples=5, save_path=None, show=True):
    indices = np.random.choice(len(results['ids']), min(num_examples, len(results['ids'])), replace=False)
    fig, axes = plt.subplots(len(indices), 2, figsize=(20, 4*len(indices)))
    if len(indices) == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        seq_id = results['ids'][idx]
        pred = results['predictions'][idx]
        label = results['labels'][idx]
        cam_left = results['cam_left'][idx].numpy()
        cam_right = results['cam_right'][idx].numpy()

        axes[i, 0].imshow(cam_left.reshape(1, -1), aspect='auto', cmap='RdBu_r')
        axes[i, 0].set_title(f'Left End CAM - {seq_id}\nPred: {pred:.3f}, Label: {int(label)}')
        axes[i, 0].set_xlabel('Position'); axes[i, 0].set_yticks([])

        axes[i, 1].imshow(cam_right.reshape(1, -1), aspect='auto', cmap='RdBu_r')
        axes[i, 1].set_title(f'Right End CAM - {seq_id}\nPred: {pred:.3f}, Label: {int(label)}')
        axes[i, 1].set_xlabel('Position'); axes[i, 1].set_yticks([])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CAM examples saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_roc_prc(y_true, y_pred, save_path=None, show=True):
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score  # type: ignore

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    prc_auc = average_precision_score(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_title("Receiver Operating Characteristic (ROC)")
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    axes[1].plot(recall, precision, label=f"AP = {prc_auc:.3f}")
    axes[1].set_title("Precision–Recall Curve")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
