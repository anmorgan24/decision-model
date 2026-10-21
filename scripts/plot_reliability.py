"""Draw a run's reliability diagrams before and after temperature scaling, side by side.

    python scripts/plot_reliability.py outputs/logit_readout_sft --split val --out reliability.png

Reads predictions_<split>.pt, written by decision_model.evaluate. Each bar is
how often answers at that confidence agree with the gold (for soft gold, the
chance a teacher sample picks the top answer); the dashed line is perfect
calibration.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from decision_model.metrics import ece, reliability_bins

BAR, LINE, INK, MUTED, GRID = "#2F6FDE", "#98A2B3", "#16181D", "#555B66", "#EEF0F3"


def panel(ax, logits, targets, labels, title):
    bins = reliability_bins(logits, targets)
    centers = [(b["lower"] + b["upper"]) / 2 for b in bins if b["count"]]
    right = [b["accuracy"] for b in bins if b["count"]]
    ax.bar(centers, right, width=0.085, color=BAR, alpha=0.85, zorder=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color=LINE, linewidth=1.5, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(color=GRID, linewidth=1, zorder=0)
    ax.set_title(f"{title}\nECE {ece(logits, targets, labels):.3f}", fontsize=13, color=INK, pad=10)
    ax.set_xlabel("confidence of top answer", fontsize=11, color=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", default="reliability.png")
    args = parser.parse_args()

    preds = torch.load(args.run_dir / f"predictions_{args.split}.pt", weights_only=False)
    if preds["calibrated_logits"] is None:
        raise SystemExit(f"{args.run_dir} has no calibrated predictions (zero-shot runs don't fit temperatures)")
    keep = preds["labels"] != -100
    targets, labels = preds["targets"][keep], preds["labels"][keep]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True)
    panel(axes[0], preds["logits"][keep], targets, labels, "Before temperature scaling")
    panel(axes[1], preds["calibrated_logits"][keep], targets, labels, "After temperature scaling")
    axes[0].set_ylabel("how often the top answer is right", fontsize=11, color=MUTED)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(Path(args.out).with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
