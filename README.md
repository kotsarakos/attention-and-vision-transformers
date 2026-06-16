# Attention Mechanisms and Transformer Networks

Coursework for **ML2 / ΕΠ34 — Machine Learning and Applications** (Harokopio
University of Athens), Assignment 2: *Attention Mechanisms and Transformer
Networks*.

The project is organised in three self-contained parts:

1. **Part 1** — A from-scratch implementation of scaled dot-product attention in
   NumPy, applied to a self-attention and a cross-attention example, and verified
   against PyTorch.
2. **Part 2** — Parameter analysis of a small encoder-only Transformer
   (BERT-style) implemented in PyTorch.
3. **Part 3** — A transfer-learning comparison of a convolutional model
   (ResNet50) and a Vision Transformer (ViT-B/16) on the Oxford-IIIT Pet dataset.

## Repository structure

```
.
├── experiment1.py            # Part 1 — scaled dot-product attention (NumPy)
├── experiment2.py            # Part 2 — Transformer encoder parameter analysis
├── experiment3.py            # Part 3 — ResNet50 vs ViT-B/16 transfer learning
├── E.npy                     # Token embedding matrix (Part 1)
├── W_Q_self.npy ...          # Projection matrices for the self/cross examples
└── vocab.json                # Vocabulary mapping for the embedding matrix
```

## Requirements

- Python 3.10 or newer
- NumPy
- Matplotlib
- PyTorch and TorchVision (Part 2 and Part 3; also used to cross-check Part 1)

A GPU is strongly recommended for Part 3, especially for full fine-tuning.

### Installation

```bash
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install numpy matplotlib torch torchvision
```

Install the PyTorch build that matches your CUDA version by following the
official instructions at https://pytorch.org if you intend to use a GPU.

## Part 1 — Scaled dot-product attention (NumPy)

Implements `scaled_dot_product_attention(Q, K, V)` using NumPy only, runs it on
a self-attention example (English sentence) and a cross-attention example
(Greek queries attending to English keys/values), prints the shape of every
intermediate quantity, verifies the result against
`torch.nn.functional.scaled_dot_product_attention`, and renders the attention
matrices as heatmaps.

```bash
python experiment1.py
```

Outputs:

- Console report of the matrix dimensions, intermediate shapes, and the
  NumPy-vs-PyTorch comparison (agreement within a 1e-5 tolerance).
- `attention_self.png` and `attention_cross.png` — attention heatmaps.

The required data files (`E.npy`, the `W_{Q,K,V}_{self,cross}.npy` projection
matrices, and `vocab.json`) are included in the repository.

## Part 2 — Transformer encoder parameter analysis

Instantiates a small encoder-only Transformer and prints, for every named
parameter, its shape and number of elements, together with the total parameter
count of the network. The script does not train the model; it is used to
analyse where the parameters come from.

```bash
python experiment2.py
```

## Part 3 — ResNet50 vs ViT-B/16 on Oxford-IIIT Pet

Loads both models pretrained on ImageNet-1K, replaces their classification heads
with a 37-class head, and compares two transfer-learning strategies:

- **Linear probing** — freeze the backbone, train only the head
  (Adam, learning rate 1e-3, 10 epochs, batch size 32, cross-entropy).
- **Full fine-tuning** — unfreeze the whole network and train with
  differentiated learning rates (small for the backbone, larger for the head).

The dataset is downloaded automatically through TorchVision on the first run.

```bash
python experiment3.py
```

### Command-line options

| Option           | Default | Description                                      |
|------------------|---------|--------------------------------------------------|
| `--epochs`       | `10`    | Number of training epochs.                       |
| `--batch-size`   | `32`    | Mini-batch size.                                 |
| `--lr-head`      | `1e-3`  | Learning rate for the classification head.       |
| `--lr-backbone`  | `1e-5`  | Backbone learning rate for full fine-tuning.     |
| `--subset`       | `None`  | Use only N samples per split (quick smoke test). |

Quick smoke test on a small subset:

```bash
python experiment3.py --epochs 1 --subset 300
```

Outputs (written to `experiment3_outputs/`):

- Per-epoch loss and accuracy curves for each run (`curves_*.png`).
- Correct and incorrect test-prediction galleries (`predictions_*.png`).
- A comparison table over all experiments (`comparison_table.txt`).

## Notes

- Part 1 runs on CPU and completes in seconds.
- Part 3 downloads roughly 800 MB of image data on the first run and benefits
  significantly from a GPU.
- All experiments use a fixed random seed (42) for reproducibility.

## Authors

- Konstantinos Kotsaras

## License and rights

Copyright (c) 2026 Konstantinos Kotsaras. All rights reserved.

This repository was developed as academic coursework for ML2 / ΕΠ34 (Machine
Learning and Applications) at Harokopio University of Athens. The source code is
provided for educational and evaluation purposes. It may not be reproduced,
redistributed, or submitted as original work by others without the explicit
permission of the author.

The Oxford-IIIT Pet dataset and the pretrained ResNet50 and ViT-B/16 weights are
the property of their respective authors and are subject to their own licenses.
