import argparse
import json
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import config
from dataset import load_csv_dataset, save_preprocessing, class_distribution
from models.models import get_model


class EMGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, correct = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct += (out.argmax(1) == yb).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate_loader(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        total_loss += loss_fn(out, yb).item() * len(yb)
        correct += (out.argmax(1) == yb).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def plot_accuracy(history: dict, feature: str, save_path) -> None:
    epochs = list(range(1, len(history["train_acc"]) + 1))
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]

    best_val = max(val_acc)
    best_epoch = val_acc.index(best_val) + 1

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        epochs, train_acc, color="#1f77b4", linewidth=2, label="Antrenare acuratețe"
    )
    ax.plot(epochs, val_acc, color="#ff7f0e", linewidth=2, label="Validare acuratețe")

    ax.scatter([best_epoch], [best_val], color="#ff7f0e", s=80, zorder=5)
    ax.annotate(
        f"Cea mai bună valoare = {best_val:.4f}\nEpoca {best_epoch}",
        xy=(best_epoch, best_val),
        xytext=(best_epoch - len(epochs) * 0.18, best_val + 0.03),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10,
    )

    ax.set_title(f"Acuratețe model ({len(epochs)} epoci)", fontsize=14)
    ax.set_xlabel("Epoci", fontsize=12)
    ax.set_ylabel("Acuratețe", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"[INFO] Grafic acuratețe salvat în: {save_path}")


def plot_accuracy_loss(history: dict, feature: str, save_path) -> None:
    epochs = list(range(1, len(history["train_acc"]) + 1))

    train_acc = history["train_acc"]
    val_acc = history["val_acc"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    best_val = max(val_acc)
    best_epoch = val_acc.index(best_val) + 1
    min_val_loss = min(val_loss)
    min_loss_epoch = val_loss.index(min_val_loss) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

     
    ax1.plot(
        epochs, train_acc, color="#1f77b4", linewidth=2, label="Antrenare acuratețe"
    )
    ax1.plot(epochs, val_acc, color="#ff7f0e", linewidth=2, label="Validare acuratețe")
    ax1.scatter([best_epoch], [best_val], color="#ff7f0e", s=80, zorder=5)
    ax1.annotate(
        f"Cea mai bună valoare = {best_val:.4f}\nEpoca {best_epoch}",
        xy=(best_epoch, best_val),
        xytext=(best_epoch - len(epochs) * 0.18, best_val + 0.03),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10,
    )
    ax1.set_title("Acuratețe model", fontsize=14)
    ax1.set_xlabel("Epoci", fontsize=12)
    ax1.set_ylabel("Acuratețe", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)

     
    ax2.plot(epochs, train_loss, color="#1f77b4", linewidth=2, label="Antrenare loss")
    ax2.plot(epochs, val_loss, color="#ff7f0e", linewidth=2, label="Validare loss")
    ax2.scatter([min_loss_epoch], [min_val_loss], color="#ff7f0e", s=80, zorder=5)
    ax2.annotate(
        f"Loss minim = {min_val_loss:.4f}\nEpoca {min_loss_epoch}",
        xy=(min_loss_epoch, min_val_loss),
        xytext=(
            min_loss_epoch - len(epochs) * 0.18,
            min_val_loss + (max(val_loss) - min_val_loss) * 0.2,
        ),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10,
    )
    ax2.set_title("Loss model", fontsize=14)
    ax2.set_xlabel("Epoci", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        f"Training model ({len(epochs)} epochs) – feature: {feature.upper()}",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"[INFO] Accuracy+loss plot saved to: {save_path}")


def main():
    ap = argparse.ArgumentParser(
        description="train on EMG_train_80.csv, validate on EMG_validation_10.csv"
    )
    ap.add_argument("--train_csv", default=str(config.TRAIN_CSV))
    ap.add_argument("--val_csv", default=str(config.VALIDATION_CSV))
    ap.add_argument(
        "--feature", default="temporal", choices=["temporal", "features", "spectral"]
    )
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--lr", type=float, default=config.LR)
    ap.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    args = ap.parse_args()

    set_seed(config.RANDOM_SEED)
    config.SAVED_MODELS_DIR.mkdir(exist_ok=True)
    config.RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print(f"Model feature: {args.feature.upper()}")
    print(f"Train CSV:      {args.train_csv}")
    print(f"Validation CSV: {args.val_csv}")
    print(f"Epochs: {args.epochs} | LR: {args.lr} | Batch: {args.batch}")
    print("=" * 70)

    X_train, y_train, train_meta, stats = load_csv_dataset(
        args.train_csv, feature=args.feature, fit_stats=True
    )
    save_preprocessing(stats)

    X_val, y_val, val_meta, _ = load_csv_dataset(
        args.val_csv, feature=args.feature, stats=stats
    )

    print("\n" + "=" * 30 + " VERIFICARE DISTRIBUȚIE " + "=" * 30)
    print(f"TRAIN STD: {np.std(X_train):.5f} | TRAIN MEAN: {np.mean(X_train):.5f}")
    print(f"VAL STD:   {np.std(X_val):.5f} | VAL MEAN:   {np.mean(X_val):.5f}")
    print("=" * 84 + "\n")

    print(f"[INFO] Train windows: {len(X_train)} | shape: {X_train.shape}")
    print(f"[INFO] Val windows:   {len(X_val)} | shape: {X_val.shape}")
    print(f"[INFO] Train class distribution: {class_distribution(y_train)}")
    print(f"[INFO] Val class distribution:   {class_distribution(y_val)}")
    print(f"[INFO] Normalizare salvată în: {config.PREPROCESSING_FILE}")

    train_loader = DataLoader(
        EMGDataset(X_train, y_train),
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        EMGDataset(X_val, y_val),
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    model = get_model(args.feature, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_name = f"emgnet_{args.feature}_best.pt"
    last_name = f"emgnet_{args.feature}_last.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        val_loss, val_acc = evaluate_loader(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"loss={train_loss:.4f} acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.SAVED_MODELS_DIR / best_name)
            print(f"  >> salvat {best_name} (val_acc={best_val_acc:.4f})")

    torch.save(model.state_dict(), config.SAVED_MODELS_DIR / last_name)

    history_path = config.RESULTS_DIR / f"history_emgnet_{args.feature}.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_path = config.RESULTS_DIR / f"grafic_acuratete_{args.feature}.png"
    plot_accuracy(history, args.feature, plot_path)

    plot_acc_loss_path = (
        config.RESULTS_DIR / f"grafic_acuratete_loss_{args.feature}.png"
    )
    plot_accuracy_loss(history, args.feature, plot_acc_loss_path)

    summary = {
        "feature": args.feature,
        "train_csv": args.train_csv,
        "validation_csv": args.val_csv,
        "train_windows": int(len(X_train)),
        "validation_windows": int(len(X_val)),
        "best_val_acc": float(best_val_acc),
        "model_best": str(config.SAVED_MODELS_DIR / best_name),
        "model_last": str(config.SAVED_MODELS_DIR / last_name),
        "preprocessing": str(config.PREPROCESSING_FILE),
        "plot_accuracy": str(plot_path),
        "plot_accuracy_loss": str(plot_acc_loss_path),
    }
    summary_path = config.RESULTS_DIR / f"training_summary_{args.feature}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[DONE] Antrenare finalizată")
    print(f"[DONE] Best val_acc: {best_val_acc:.4f}")
    print(f"[DONE] Model best: {config.SAVED_MODELS_DIR / best_name}")
    print(f"[DONE] History: {history_path}")
    print(f"[DONE] Grafic: {plot_path}")
    print(f"[DONE] Grafic acuratețe+loss: {plot_acc_loss_path}")
    print(f"[DONE] Summary: {summary_path}")


if __name__ == "__main__":
    main()
