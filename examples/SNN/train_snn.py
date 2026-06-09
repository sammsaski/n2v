"""Train an F2FMLP SNN on MNIST.

Architecture: 784 -> 128 -> 64 -> 10, T=16 timesteps (latency coding)

Saves models/mnist_snn.pt (state dict) and models/mnist_snn_meta.json.
Use verify_snn.py to run reachability verification on the trained model.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from n2v.snn.model import F2FMLP
from n2v.snn.encoding import encode_batch

SCRIPT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
INPUT_SIZE   = 784
HIDDEN_SIZES = [128, 64]
NUM_CLASSES  = 10
NUM_STEPS    = 16
BETA         = 0.9
THRESHOLD    = 1.0

BATCH_SIZE   = 128
NUM_EPOCHS   = 10
LR           = 1e-3
SEED         = 42


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for images, labels in loader:
        images = images.view(images.size(0), -1).to(device)   # (B, 784)
        labels = labels.to(device)
        spikes  = encode_batch(images, model.num_steps)        # (B, 784, T)
        scores  = model(spikes)                                # (B, 10)
        loss    = criterion(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (scores.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images = images.view(images.size(0), -1).to(device)
        spikes = encode_batch(images, model.num_steps)
        scores = model(spikes)
        correct += (scores.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ---- Data ----------------------------------------------------------------
    print("\nLoading MNIST...")
    tf = transforms.ToTensor()
    data_dir = str(SCRIPT_DIR / "data")
    train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  Train: {len(train_ds):,}   Test: {len(test_ds):,}")

    # ---- Model ---------------------------------------------------------------
    model = F2FMLP(
        input_size   = INPUT_SIZE,
        hidden_sizes = HIDDEN_SIZES,
        num_classes  = NUM_CLASSES,
        beta         = BETA,
        threshold    = THRESHOLD,
        num_steps    = NUM_STEPS,
    ).to(device)
    print(f"\nArchitecture: F2FMLP  784 -> {' -> '.join(str(h) for h in HIDDEN_SIZES)} -> {NUM_CLASSES}")
    print(f"              T={NUM_STEPS} timesteps, beta={BETA}, theta={THRESHOLD}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters   : {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ---- Training ------------------------------------------------------------
    print(f"\nTraining for {NUM_EPOCHS} epochs (lr={LR})...\n")
    t0 = time.time()
    history = []
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = eval_epoch(model, test_loader, device)
        history.append({"epoch": epoch, "loss": train_loss, "train_acc": train_acc, "test_acc": test_acc})
        print(f"  Epoch {epoch:2d}/{NUM_EPOCHS}  loss={train_loss:.4f}  "
              f"train={train_acc:.1f}%  test={test_acc:.1f}%")

    total_time = time.time() - t0
    final_acc  = history[-1]["test_acc"]
    print(f"\nFinal test accuracy : {final_acc:.1f}%")
    print(f"Training time       : {total_time:.1f}s")

    # ---- Save ----------------------------------------------------------------
    models_dir = SCRIPT_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    model.eval()
    # Save state dict rather than the full object: snntorch's fast_sigmoid
    # activation is defined as a nested closure which pickle cannot serialize.
    torch.save(model.state_dict(), models_dir / "mnist_snn.pt")
    with open(models_dir / "mnist_snn_meta.json", "w") as f:
        json.dump({
            "input_size"  : INPUT_SIZE,
            "hidden_sizes": HIDDEN_SIZES,
            "num_classes" : NUM_CLASSES,
            "num_steps"   : NUM_STEPS,
            "beta"        : BETA,
            "threshold"   : THRESHOLD,
            "test_acc"    : final_acc,
            "epochs"      : NUM_EPOCHS,
            "history"     : history,
        }, f, indent=2)

    print("\nSaved:")
    print(f"  {models_dir / 'mnist_snn.pt'}       <- state dict")
    print(f"  {models_dir / 'mnist_snn_meta.json'} <- training summary")
    print("\nNext: run verify_snn.py")
