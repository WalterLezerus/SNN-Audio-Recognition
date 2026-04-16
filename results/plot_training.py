import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS = "C:/Users/erick/Documents/ResearchProjects/AudioRecognitionWithSNN/results"

summary = pd.read_csv(f"{RESULTS}/epoch_summary.csv")
steps = pd.read_csv(f"{RESULTS}/training_steps.csv")

plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {"train": "#4C9BE8", "val": "#E8734C", "loss": "#6DBE6D", "time": "#B07DD9"}

# --- Figure 1: Learning Curve (train vs val acc per epoch) ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(summary["epoch"], summary["train_acc"], marker="o", color=COLORS["train"], label="Train Acc")
ax.plot(summary["epoch"], summary["val_acc"], marker="o", color=COLORS["val"], label="Val Acc")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Learning Curve — SNN Audio Recognition (36 Classes)")
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax.legend()
ax.annotate(f"Best val: {summary['val_acc'].max():.1%}",
            xy=(summary["val_acc"].idxmax() + 1, summary["val_acc"].max()),
            xytext=(10, -18), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9)
fig.tight_layout()
fig.savefig(f"{RESULTS}/fig1_learning_curve.png", dpi=150)
print("Saved fig1_learning_curve.png")

# --- Figure 2: Loss Curve ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(summary["epoch"], summary["loss"], marker="o", color=COLORS["loss"])
ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss")
ax.set_title("Loss Curve — SNN Audio Recognition (36 Classes)")
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
fig.tight_layout()
fig.savefig(f"{RESULTS}/fig2_loss_curve.png", dpi=150)
print("Saved fig2_loss_curve.png")

# --- Figure 3: Efficiency Curve (cumulative time vs val acc) ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(summary["total_time_min"], summary["val_acc"], marker="o", color=COLORS["val"])
for _, row in summary.iterrows():
    ax.annotate(f"E{int(row['epoch'])}", xy=(row["total_time_min"], row["val_acc"]),
                xytext=(4, 4), textcoords="offset points", fontsize=7, color="gray")
ax.set_xlabel("Cumulative Training Time (min)")
ax.set_ylabel("Val Accuracy")
ax.set_title("Training Efficiency — Val Accuracy vs. Total Time")
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
fig.tight_layout()
fig.savefig(f"{RESULTS}/fig3_efficiency_curve.png", dpi=150)
print("Saved fig3_efficiency_curve.png")

# --- Figure 4: Intra-epoch accuracy (% complete as x-axis, colored by epoch) ---
fig, ax = plt.subplots(figsize=(11, 6))
cmap = plt.cm.viridis
epochs = steps["epoch"].unique()
norm = plt.Normalize(epochs.min(), epochs.max())
for ep in epochs:
    ep_data = steps[steps["epoch"] == ep]
    ax.plot(ep_data["pct_complete"], ep_data["train_acc"],
            color=cmap(norm(ep)), alpha=0.85, linewidth=1.5)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label="Epoch")
ax.set_xlabel("Epoch Completion (%)")
ax.set_ylabel("Train Accuracy")
ax.set_title("Intra-Epoch Accuracy by Epoch — SNN Audio Recognition")
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
fig.tight_layout()
fig.savefig(f"{RESULTS}/fig4_intraepoch_accuracy.png", dpi=150)
print("Saved fig4_intraepoch_accuracy.png")

print("\nAll figures saved to", RESULTS)
