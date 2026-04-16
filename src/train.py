# Training loop for SNNAudioNet
# Same structure as gesture project: Adam + StepLR + CrossEntropyLoss + checkpoint on best val acc

from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from dataset import get_dataloaders, NUM_CLASSES
from model import SNNAudioNet

CHECKPOINT_DIR = Path(__file__).parent.parent / "models"
LOG_DIR        = Path(__file__).parent.parent / "models"

EPOCHS      = 30
BATCH_SIZE  = 32
LR          = 1e-3
N_TIME_STEPS = 50  # rate-coded spike frames per sample


def train(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, n_time_steps=N_TIME_STEPS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders(
        batch_size=batch_size, n_time_steps=n_time_steps
    )

    model     = SNNAudioNet(num_classes=NUM_CLASSES).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    loss_fn   = nn.CrossEntropyLoss()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_path, "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"Training on: {device}")
    log(f"Classes: {NUM_CLASSES}  |  Time steps: {n_time_steps}  |  Batch: {batch_size}")
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct    = 0
        total      = 0

        for data, targets in train_loader:
            # data: (n_time_steps, batch, 1, N_MELS, T) -- already transposed by collate_fn
            data    = data.float().to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(data)
            loss   = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds    = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total   += targets.size(0)

        train_acc = correct / total
        val_acc   = evaluate(model, val_loader, device)
        scheduler.step()

        log(
            f"Epoch {epoch:>3}/{epochs} | "
            f"Loss: {total_loss / len(train_loader):.4f} | "
            f"Train acc: {train_acc:.3f} | "
            f"Val acc: {val_acc:.3f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best.pth")
            log(f"  Saved checkpoint (val acc: {best_acc:.3f})")

    log(f"\nDone. Best val acc: {best_acc:.3f}")
    log_file.close()


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for data, targets in loader:
            data    = data.float().to(device)
            targets = targets.to(device)
            logits  = model(data)
            preds   = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total   += targets.size(0)
    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    train()
