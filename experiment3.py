"""
Part 3 — Comparison of ResNet50 (CNN) and ViT-B/16 (Transformer) on the
Oxford-IIIT Pet Dataset (37 breeds of cats and dogs).

Both models are loaded pretrained on ImageNet-1K and adapted to the 37-class
problem with two transfer-learning strategies:

    1. Linear probing  : freeze the backbone, train only the new classifier
                          head (Adam, lr=1e-3, 10 epochs, batch=32, CE loss).
    2. Full fine-tuning : unfreeze everything and train with DIFFERENTIATED
                          learning rates (small lr for the backbone, larger lr
                          for the head), same batch size / epochs.

For every (model, strategy) combination the script reports:
    - train / validation / test accuracy,
    - per-epoch loss and accuracy curves (saved as PNG),
    - number of trainable parameters and average time per epoch.

It also:
    - visualises correct and incorrect test predictions with the probability
      assigned to the correct and to the predicted (wrong) class,
    - prints a final comparison table over the four experiments.

Run with (GPU strongly recommended for full fine-tuning):
    python experiment3.py
Optional flags allow a quick smoke test, e.g.:
    python experiment3.py --epochs 1 --subset 200
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms as T
from torchvision.datasets import OxfordIIITPet
from torchvision.models import (
    ResNet50_Weights,
    ViT_B_16_Weights,
    resnet50,
    vit_b_16,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "experiment3_outputs"
NUM_CLASSES = 37
SEED = 42


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_transforms(weights, strong: bool = False):
    """
    Build (train, eval) transforms for a given set of pretrained weights.

    The eval transform is the one recommended by the weights (the exact
    preprocessing used during the model's own pretraining). The train
    transform reuses the same crop size and normalisation but adds data
    augmentation:
        - strong=False : light augmentation (random resized crop + flip),
                         used for the four baseline experiments.
        - strong=True  : heavier augmentation (more aggressive crop,
                         RandAugment, colour jitter, random erasing), used for
                         the two extra fine-tuning experiments that test
                         whether stronger augmentation improves generalisation.
    """
    eval_tf = weights.transforms()
    crop_size = eval_tf.crop_size[0]
    mean, std = eval_tf.mean, eval_tf.std

    if strong:
        train_tf = T.Compose([
            T.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandAugment(),
            T.ColorJitter(0.3, 0.3, 0.3),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(p=0.25),
        ])
    else:
        train_tf = T.Compose([
            T.RandomResizedCrop(crop_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return train_tf, eval_tf


def build_loaders(weights, batch_size: int, subset: int | None = None,
                  strong: bool = False):
    """
    Create train / val / test dataloaders for the model with the given weights.

    - trainval split is divided 80% / 20% into train / validation using
      random_split with a fixed seed.
    - The training subset uses augmentation (light or strong, see
      build_transforms); validation and test use the deterministic eval
      transform.
    - Because train and val need different transforms, two dataset instances
      over the same trainval split are created and indexed with the same
      random partition.
    """
    train_tf, eval_tf = build_transforms(weights, strong=strong)

    train_full = OxfordIIITPet(DATA_DIR, split="trainval", target_types="category",
                               transform=train_tf, download=True)
    val_full = OxfordIIITPet(DATA_DIR, split="trainval", target_types="category",
                             transform=eval_tf, download=True)
    test_set = OxfordIIITPet(DATA_DIR, split="test", target_types="category",
                             transform=eval_tf, download=True)

    n = len(train_full)
    n_val = int(0.2 * n)
    n_train = n - n_val
    gen = torch.Generator().manual_seed(SEED)
    train_part, val_part = random_split(range(n), [n_train, n_val], generator=gen)

    train_set = Subset(train_full, train_part.indices)
    val_set = Subset(val_full, val_part.indices)

    # Optional: shrink everything for a quick smoke test.
    if subset is not None:
        train_set = Subset(train_set, list(range(min(subset, len(train_set)))))
        val_set = Subset(val_set, list(range(min(subset, len(val_set)))))
        test_set = Subset(test_set, list(range(min(subset, len(test_set)))))

    common = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def build_resnet50():
    """ResNet50 pretrained on ImageNet-1K with a fresh 37-way head."""
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model, weights, model.fc


def build_vit():
    """ViT-B/16 pretrained on ImageNet-1K with a fresh 37-way head."""
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    return model, weights, model.heads.head


def set_requires_grad(model: nn.Module, flag: bool) -> None:
    for p in model.parameters():
        p.requires_grad = flag


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, criterion) -> tuple[float, float]:
    """Return (average loss, accuracy) of the model over a dataloader."""
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def train_one_epoch(model, loader, device, criterion, optimizer) -> tuple[float, float]:
    """Train for a single epoch; return (average loss, accuracy)."""
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)


@dataclass
class Result:
    model_name: str
    strategy: str
    trainable_params: int
    train_acc: float
    val_acc: float
    test_acc: float
    time_per_epoch: float
    history: History


@dataclass
class ExpConfig:
    """One experiment specification."""
    model_name: str
    strategy: str
    optimizer_kind: str           # "probe" or "finetune"
    strong_aug: bool = False
    epochs: int | None = None     # None -> use the default (--epochs)
    scheduler: bool = False       # use warmup + cosine LR schedule


def run_training(model_name, strategy, model, optimizer, loaders, device,
                 epochs, scheduler=None) -> Result:
    """Full training loop for one (model, strategy). Returns a Result."""
    train_loader, val_loader, test_loader = loaders
    criterion = nn.CrossEntropyLoss()
    hist = History()

    print(f"\n=== {model_name} | {strategy} "
          f"| trainable params = {count_trainable(model):,} ===")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, criterion, optimizer)
        dt = time.time() - t0
        va_loss, va_acc = evaluate(model, val_loader, device, criterion)

        if scheduler is not None:
            scheduler.step()

        hist.train_loss.append(tr_loss)
        hist.train_acc.append(tr_acc)
        hist.val_loss.append(va_loss)
        hist.val_acc.append(va_acc)
        hist.epoch_times.append(dt)

        print(f"  epoch {epoch:2d}/{epochs}  "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}  ({dt:.1f}s)")

    # Final accuracies on all three splits.
    _, train_acc = evaluate(model, train_loader, device, criterion)
    _, val_acc = evaluate(model, val_loader, device, criterion)
    _, test_acc = evaluate(model, test_loader, device, criterion)
    print(f"  FINAL  train_acc={train_acc:.4f}  "
          f"val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")

    return Result(
        model_name=model_name,
        strategy=strategy,
        trainable_params=count_trainable(model),
        train_acc=train_acc,
        val_acc=val_acc,
        test_acc=test_acc,
        time_per_epoch=float(np.mean(hist.epoch_times)),
        history=hist,
    )


# ---------------------------------------------------------------------------
# Optimizer builders for the two strategies
# ---------------------------------------------------------------------------

def make_linear_probe_optimizer(model, head, lr=1e-3):
    """
    Linear probing: freeze the whole backbone, train only the head.
    Adam over the head parameters with lr = 1e-3.
    """
    set_requires_grad(model, False)
    set_requires_grad(head, True)
    return torch.optim.Adam(head.parameters(), lr=lr)


def make_finetune_optimizer(model, head, lr_backbone=1e-5, lr_head=1e-3):
    """
    Full fine-tuning: unfreeze everything, but use DIFFERENTIATED learning rates
    """
    set_requires_grad(model, True)
    head_ids = {id(p) for p in head.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return torch.optim.Adam([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head.parameters(), "lr": lr_head},
    ])


def make_scheduler(optimizer, epochs, warmup_epochs=2):
    """
    Learning-rate schedule for the improved fine-tuning experiment: a short
    linear warmup (which protects the pretrained backbone during the first
    epochs) followed by cosine annealing of the learning rate down towards 0
    (smooth convergence in the final epochs). Stepped once per epoch.
    """
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs))
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_history(result: Result, out_path: Path) -> None:
    """Save per-epoch loss and accuracy curves (train vs validation)."""
    h = result.history
    epochs = range(1, len(h.train_loss) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax_loss.plot(epochs, h.train_loss, "o-", label="train")
    ax_loss.plot(epochs, h.val_loss, "s-", label="validation")
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("loss")
    ax_loss.set_title("Loss"); ax_loss.legend(); ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, h.train_acc, "o-", label="train")
    ax_acc.plot(epochs, h.val_acc, "s-", label="validation")
    ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Accuracy"); ax_acc.legend(); ax_acc.grid(alpha=0.3)

    fig.suptitle(f"{result.model_name} — {result.strategy}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved curves: {out_path.name}")


# ---------------------------------------------------------------------------
# Visualisation of correct / incorrect predictions
# ---------------------------------------------------------------------------

def denormalize(img: torch.Tensor, mean, std) -> np.ndarray:
    """Undo normalisation and return an (H, W, C) array in [0, 1] for display."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    img = img.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


@torch.no_grad()
def visualize_predictions(model, weights, test_loader, device, classes,
                          out_path: Path, n_correct=4, n_wrong=4) -> None:
    """
    Collect a few correctly and a few incorrectly classified test images and
    plot them with the probability of the true and of the predicted class.
    """
    model.eval()
    eval_tf = weights.transforms()
    mean, std = eval_tf.mean, eval_tf.std

    # The test split is ordered by class, so iterating it directly would yield
    # examples from a single breed. Iterate it in shuffled order (fixed seed for
    # reproducibility) and pick correct / incorrect samples from DISTINCT
    # classes, so the figure shows a variety of animals.
    g = torch.Generator().manual_seed(SEED)
    vis_loader = DataLoader(test_loader.dataset, batch_size=test_loader.batch_size,
                            shuffle=True, num_workers=2, generator=g)

    correct_samples, wrong_samples = [], []
    seen_correct, seen_wrong = set(), set()
    for x, y in vis_loader:
        probs = torch.softmax(model(x.to(device)), dim=1).cpu()
        preds = probs.argmax(1)
        for i in range(x.size(0)):
            true_c, pred_c = int(y[i]), int(preds[i])
            sample = (x[i], true_c, pred_c, float(probs[i, true_c]), float(probs[i, pred_c]))
            if pred_c == true_c:
                if len(correct_samples) < n_correct and true_c not in seen_correct:
                    correct_samples.append(sample)
                    seen_correct.add(true_c)
            else:
                if len(wrong_samples) < n_wrong and true_c not in seen_wrong:
                    wrong_samples.append(sample)
                    seen_wrong.add(true_c)
        if len(correct_samples) >= n_correct and len(wrong_samples) >= n_wrong:
            break

    samples = correct_samples + wrong_samples
    cols = max(n_correct, n_wrong)
    fig, axes = plt.subplots(2, cols, figsize=(3.4 * cols, 9.0))

    # Hide every axis first (so empty slots stay blank).
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for j, (img, true_c, pred_c, p_true, p_pred) in enumerate(samples):
        row = 0 if j < n_correct else 1
        col = j if j < n_correct else j - n_correct
        ax = axes[row, col]
        ax.imshow(denormalize(img, mean, std))
        ax.set_xticks([])
        ax.set_yticks([])

        ok = pred_c == true_c
        color = "green" if ok else "red"
        # Coloured frame around the image (green=correct, red=wrong).
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)

        title = (f"true: {classes[true_c]}\n"
                 f"pred: {classes[pred_c]}\n"
                 f"p(true)={p_true:.2f}   p(pred)={p_pred:.2f}")
        ax.set_title(title, color=color, fontsize=9, pad=6)

    # Row group labels on the far left.
    axes[0, 0].set_ylabel("CORRECT", color="green", fontsize=13,
                          fontweight="bold", labelpad=10)
    axes[1, 0].set_ylabel("WRONG", color="red", fontsize=13,
                          fontweight="bold", labelpad=10)

    fig.suptitle("Test predictions (top: correctly classified, "
                 "bottom: misclassified)", fontsize=13)
    # Leave generous vertical room so the bottom-row titles never overlap
    # the top-row images.
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.04,
                        wspace=0.08, hspace=0.45)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved predictions: {out_path.name}")


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(results: list[Result]) -> None:
    print("\n" + "=" * 96)
    print("Comparison of all experiments")
    print("=" * 96)
    header = (f"{'Model':<10} {'Strategy':<26} "
              f"{'Trainable params':>18} {'Test acc':>10} {'Time/epoch (s)':>16}")
    print(header)
    print("-" * 96)
    for r in results:
        print(f"{r.model_name:<10} {r.strategy:<26} "
              f"{r.trainable_params:>18,} {r.test_acc:>10.4f} {r.time_per_epoch:>16.1f}")
    print("=" * 96)

    # Also save it as a plain text file for the report.
    out = OUT_DIR / "comparison_table.txt"
    with out.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for r in results:
            f.write(f"{r.model_name:<10} {r.strategy:<26} "
                    f"{r.trainable_params:>18,} {r.test_acc:>10.4f} "
                    f"{r.time_per_epoch:>16.1f}\n")
    print(f"saved table: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-backbone", type=float, default=1e-5,
                   help="backbone learning rate for full fine-tuning")
    p.add_argument("--subset", type=int, default=None,
                   help="use only N samples per split (quick smoke test)")
    return p.parse_args()


def get_class_names(loader) -> list[str]:
    """Recover human-readable class names from a (possibly nested) Subset."""
    base = loader.dataset
    while hasattr(base, "dataset"):
        base = base.dataset
    return base.classes


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    device = get_device()
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Device: {device}")

    builders = {"ResNet50": build_resnet50, "ViT-B/16": build_vit}

    # Experiment plan:
    #   - 4 baseline experiments (2 models x 2 strategies),
    #   - 2 extra fine-tuning runs with stronger augmentation,
    #   - 2 improved fine-tuning runs (augmentation + warmup/cosine LR schedule
    #     + more epochs, for both models) that answer "what would you change to
    #     improve fine-tuning?".
    experiments = [
        ExpConfig("ResNet50", "linear probing",   "probe"),
        ExpConfig("ResNet50", "full fine-tuning", "finetune"),
        ExpConfig("ViT-B/16", "linear probing",   "probe"),
        ExpConfig("ViT-B/16", "full fine-tuning", "finetune"),
        ExpConfig("ResNet50", "fine-tuning + aug", "finetune", strong_aug=True),
        ExpConfig("ViT-B/16", "fine-tuning + aug", "finetune", strong_aug=True),
        ExpConfig("ResNet50", "fine-tuning + aug + sched", "finetune",
                  strong_aug=True, epochs=30, scheduler=True),
        ExpConfig("ViT-B/16", "fine-tuning + aug + sched", "finetune",
                  strong_aug=True, epochs=30, scheduler=True),
    ]

    # Cache dataloaders by (model, strong) so they are built only once.
    loader_cache: dict[tuple[str, bool], tuple] = {}
    results: list[Result] = []

    for cfg in experiments:
        builder = builders[cfg.model_name]
        epochs = cfg.epochs if cfg.epochs is not None else args.epochs

        key = (cfg.model_name, cfg.strong_aug)
        if key not in loader_cache:
            _, weights, _ = builder()
            loader_cache[key] = build_loaders(weights, args.batch_size,
                                              subset=args.subset, strong=cfg.strong_aug)
        loaders = loader_cache[key]
        class_names = get_class_names(loaders[2])

        set_seed(SEED)  # identical initial conditions per experiment
        model, weights, head = builder()
        model.to(device)

        if cfg.optimizer_kind == "probe":
            optimizer = make_linear_probe_optimizer(model, head, lr=args.lr_head)
        else:
            optimizer = make_finetune_optimizer(
                model, head, lr_backbone=args.lr_backbone, lr_head=args.lr_head)
        scheduler = make_scheduler(optimizer, epochs) if cfg.scheduler else None

        result = run_training(cfg.model_name, cfg.strategy, model, optimizer,
                              loaders, device, epochs, scheduler=scheduler)
        results.append(result)

        tag = (f"{cfg.model_name}_{cfg.strategy}"
               .replace("/", "").replace("-", "").replace("+", "plus")
               .replace(" ", "_"))
        plot_history(result, OUT_DIR / f"curves_{tag}.png")
        visualize_predictions(model, weights, loaders[2], device, class_names,
                              OUT_DIR / f"predictions_{tag}.png")

    print_comparison_table(results)


if __name__ == "__main__":
    main()
