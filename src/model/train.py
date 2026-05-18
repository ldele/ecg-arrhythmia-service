"""Training loop for the ECG 1D CNN."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from src.model.architecture import ECGNet
from src.model.dataset import N_CLASSES, EcgBeatDataset

CLASS_NAMES = ["N", "S", "V", "F"]


@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.1
    seed: int = 42


def _compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights for the cross-entropy loss."""
    counts = Counter(y.tolist())
    n = len(y)
    weights = np.array(
        [n / (N_CLASSES * counts[c]) for c in range(N_CLASSES)],
        dtype=np.float32,
    )
    return torch.from_numpy(weights)


def _split_train_val(
    dataset: EcgBeatDataset, val_fraction: float, seed: int
) -> tuple[
    torch.utils.data.Subset[tuple[torch.Tensor, torch.Tensor]],
    torch.utils.data.Subset[tuple[torch.Tensor, torch.Tensor]],
]:
    n = len(dataset)
    n_val = int(n * val_fraction)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return (
        torch.utils.data.Subset(dataset, train_idx),
        torch.utils.data.Subset(dataset, val_idx),
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                assert optimizer is not None
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(-1) == y).sum().item()
            total += x.size(0)

    return total_loss / total, total_correct / total


def train(
    train_dataset: EcgBeatDataset,
    test_dataset: EcgBeatDataset,
    config: TrainConfig,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train_subset, val_subset = _split_train_val(
        train_dataset, config.val_fraction, config.seed
    )
    train_loader = DataLoader(
        train_subset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_subset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    model = ECGNet(n_classes=N_CLASSES).to(device)
    # weights = _compute_class_weights(train_dataset.y).to(device)
    criterion = nn.CrossEntropyLoss() # criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    best_val_acc = 0.0
    for epoch in range(1, config.epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, None, device)
        print(
            f"epoch {epoch:02d} | train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / "model.pt")

    # Final evaluation on the held-out DS2 set with the best checkpoint.
    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=device))
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_targets.extend(y.tolist())

    report = classification_report(
        all_targets, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(N_CLASSES)))

    metrics = {
        "test_classification_report": report,
        "test_confusion_matrix": cm.tolist(),
        "config": config.__dict__,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n--- DS2 (test) report ---")
    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)