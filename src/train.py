# Training loop for SNNAudioNet
# Same structure as gesture project: Adam + StepLR + CrossEntropyLoss + checkpoint on best val acc

from pathlib import Path
from datetime import datetime
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch_directml

from dataset import get_dataloaders, NUM_CLASSES
from model import SNNAudioNet

CHECKPOINT_DIR = Path(__file__).parent.parent / "models"
LOG_DIR        = Path(__file__).parent.parent / "models"

EPOCHS      = 15
BATCH_SIZE  = 32
LR          = 1e-3
N_TIME_STEPS = 50  # rate-coded spike frames per sample


RESUME_CHECKPOINT = CHECKPOINT_DIR / "resume.pth"


def train(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, n_time_steps=N_TIME_STEPS, resume=False):
    device = torch_directml.device()

    train_loader, val_loader = get_dataloaders(
        batch_size=batch_size, n_time_steps=n_time_steps
    )

    model     = SNNAudioNet(num_classes=NUM_CLASSES).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    loss_fn   = nn.CrossEntropyLoss()

    start_epoch = 1
    best_acc    = 0.0

    if resume and RESUME_CHECKPOINT.exists():
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc    = checkpoint["best_acc"]
        print(f"Resumed from epoch {checkpoint['epoch']} (best val acc: {best_acc:.3f})")
    elif resume:
        print("No resume checkpoint found -- starting from scratch.")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_path, "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"Training on: {device}")
    log(f"Classes: {NUM_CLASSES}  |  Time steps: {n_time_steps}  |  Batch: {batch_size}")
    log(f"Starting from epoch {start_epoch}/{epochs}")
    run_start  = time.time()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss  = 0.0
        correct     = 0
        total       = 0
        epoch_start = time.time()
        n_batches   = len(train_loader)
        milestones  = {int(n_batches * p) for p in (0.1, 0.2, 0.3, 0.4, 0.5,
                                                      0.6, 0.7, 0.8, 0.9, 1.0)}

        for batch_idx, (data, targets) in enumerate(train_loader, 1):
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

            if batch_idx in milestones:
                pct     = int(batch_idx / n_batches * 100)
                elapsed = time.time() - epoch_start
                print(f"  Epoch {epoch} {pct:>3}% | "
                      f"elapsed: {elapsed:.0f}s | "
                      f"loss: {total_loss / batch_idx:.4f} | "
                      f"acc: {correct / total:.3f}",
                      flush=True)

        train_acc    = correct / total
        epoch_time   = time.time() - epoch_start
        total_elapsed = time.time() - run_start
        val_acc      = evaluate(model, val_loader, device)
        scheduler.step()

        log(
            f"Epoch {epoch:>3}/{epochs} | "
            f"Loss: {total_loss / n_batches:.4f} | "
            f"Train acc: {train_acc:.3f} | "
            f"Val acc: {val_acc:.3f} | "
            f"Epoch time: {epoch_time/60:.1f}m | "
            f"Total: {total_elapsed/60:.1f}m"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best.pth")
            log(f"  Saved best checkpoint (val acc: {best_acc:.3f})")

        # Always save full resume checkpoint
        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc":  best_acc,
        }, RESUME_CHECKPOINT)

    log(f"\nDone. Best val acc: {best_acc:.3f} | Total time: {(time.time() - run_start)/60:.1f}m")
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
    import sys
    resume = "--resume" in sys.argv
    train(resume=resume)
