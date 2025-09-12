# IR Toolkit

**A comprehensive toolkit for intron retention training and analysis with custom backbones**

IR Toolkit provides a complete pipeline for building, training, and analyzing deep learning models that predict intron retention events from genomic sequences. It features Class Activation Mapping (CAM) for interpretability, support for multiple backbone architectures including SpliceAI, and flexible training configurations.

## Key Features

- **End-to-end Pipeline**: From sequence data to trained models with interpretable predictions
- **CAM Interpretability**: Built-in Class Activation Mapping for understanding model decisions
- **Multiple Backbones**: Support for custom CNN, ParNet, and SpliceAI architectures
- **PyTorch Lightning**: Modern training with automatic mixed precision, checkpointing, and logging
- **Rich Visualizations**: Training curves, CAM heatmaps, and performance metrics
- **Flexible Configuration**: Attention pooling, joint classifiers, and middle vector fusion

## Installation

<!-- ### Basic Installation

```bash
pip install ir_toolkit
```

### Full Installation with Training Support

```bash
pip install "ir_toolkit[all]"
``` -->

### Development Installation

```bash
git clone https://github.com/melonheader/ir_toolkit.git
cd ir_toolkit
pip install -e ".[all]"
```

### Optional: SpliceAI Support

For SpliceAI backbone models, install OpenSpliceAI manually:

```bash
git clone https://github.com/Kuanhao-Chao/OpenSpliceAI.git
cd OpenSpliceAI
python setup.py install
```

## 📖 Quick Start

### 1. Prepare Your Data

```python
import ir_toolkit as irt
import pandas as pd

# Load and prepare dataset
df = irt.prepare_dataset(
    source="intron_data.csv",  # CSV with EVENT, PSI columns
    class_col="PSI",           # Column for classification
    class_lims=(0.1, 0.9),     # Low/high thresholds
    drop_mid=True              # Remove intermediate values
)

# Load sequences
sequences = irt.load_fasta_to_dict("sequences.fasta")
```

### 2. Create Dataset and DataLoaders

```python
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Split data
train_ids, val_ids = train_test_split(list(df.index), test_size=0.2)

# Create datasets
train_ds = irt.IntronsEndsDataset(
    ids=train_ids,
    fasta_seqs=sequences,
    df=df,
    L_left=512,   # Left context length
    L_right=512   # Right context length
)
val_ds = irt.IntronsEndsDataset(ids=val_ids, fasta_seqs=sequences, df=df, L_left=512, L_right=512)

# Create dataloaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, 
                         collate_fn=irt.introns_collate_fn)
val_loader = DataLoader(val_ds, batch_size=32, collate_fn=irt.introns_collate_fn)
```

### 3. Train a Model

```python
# Setup training components
trainer, model, log_collector, checkpoint = irt.setup_training(
    train_loader=train_loader,
    backbone_model=None,  # Uses default CNN backbone
    use_attention_pooling=True,
    lr=1e-4,
    max_epochs=50
)

# Run training
metrics = irt.run_training(trainer, model, train_loader, val_loader)

# Visualize training progress
irt.plot_training_history(log_collector, save_path="training_curves.png")
```

### 4. Evaluate and Interpret

```python
# Evaluate model
results = irt.evaluate_model(model, val_loader)

# Plot ROC and Precision-Recall curves
irt.plot_roc_prc(results['labels'], results['predictions'], 
                 save_path="performance_curves.png")

# Visualize CAM examples
irt.visualize_cam_examples(results, sequences, num_examples=5,
                          save_path="cam_examples.png")
```

## Architecture Overview

### Model Components

- **Backbone**: Feature extraction from one-hot encoded sequences
  - Simple CNN (default)
  - Backbone wrapper for [parnet](https://github.com/mhorlacher/parnet)
  - Backbone wrapper for [SpliceAI](https://github.com/Kuanhao-Chao/OpenSpliceAI/tree/main)
  
- **CAM Heads**: Per-position [Class Activation Mapping (CAM)](https://arxiv.org/abs/1512.04150)
  - Global Average Pooling or Attention-based pooling
  - Feature importance visualization
  
- **Fusion Strategies**:
  - Simple averaging
  - Linear fusion layer
  - Joint MLP classifier

### Training Features

- **Layer-wise Learning Rate Decay (LLRD)**: Different learning rates for different backbone layers
- **Staged Training**: Option to freeze backbone initially, then unfreeze
- **Automatic Mixed Precision**: Faster training on modern GPUs
- **Early Stopping & Checkpointing**: Robust training management

## Advanced Usage

### Using SpliceAI Backbone

```python
# Load pre-trained SpliceAI model
spliceai_model = irt.load_spliceAI_model(
    weights_path="spliceai_weights.pth",
    flanking_size=400
)

# Wrap as backbone
backbone = irt.SpliceAIBackbone(spliceai_model, cropping=True)

# Use in training
trainer, model, _, _ = irt.setup_training(
    train_loader=train_loader,
    backbone_model=backbone,
    feat_dim=32  # SpliceAI feature dimension
)
```

## 📊 Data Format

### Input CSV Structure
```
EVENT,PSI,other_features...
intron_001,0.85,value1
intron_002,0.12,value2
...
```

### FASTA Sequences
```
>intron_001
ATCGATCGATCG...
>intron_002
GCTAGCTAGCTA...
```

## 🛠️ Configuration Options

### Model Architecture
- `use_attention_pooling`: Use attention-based sequence pooling
- `use_joint_classifier`: MLP fusion of left/right/middle predictions
- `use_simple_fusion`: Linear fusion layer
- `use_middle`: Include middle region embeddings

### Training Parameters
- `lr`: Learning rate for heads/fusion layers
- `backbone_lr_base`: Base learning rate for backbone
- `llrd_gamma`: Layer-wise LR decay factor
- `freeze_backbone_initial`: Start with frozen backbone
- `unfreeze_after_epochs`: When to unfreeze backbone

## 📈 Model Interpretability

IR Toolkit provides built-in interpretability through Class Activation Mapping:

- **Per-position importance**: Visualize which sequence positions contribute to predictions
- **Left/right end analysis**: Separate analysis of 5' and 3' splice sites
- **Feature importance weights**: Extract learned feature importance from GAP classifier